"""用现有 Tokenizer 编码完整语料，按特殊 token 边界分块，保存可 mmap 的 uint16 NPY。"""

import argparse
import hashlib
import json
import os
import shutil
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np

from experiment_tokenizer_a import SEPARATOR, load_tokenizer


SEPARATOR_BYTES = SEPARATOR.encode("utf-8")
DTYPE = np.dtype("<u2")
TOKENIZER = None
SPECIAL_ID = None


def hash_prefix(path: Path, length: int | None = None):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        remaining = length
        while remaining is None or remaining > 0:
            block = source.read(8 * 1024**2 if remaining is None else min(8 * 1024**2, remaining))
            if not block:
                if remaining:
                    raise ValueError(f"文件短于检查点记录：{path}")
                break
            digest.update(block)
            if remaining is not None:
                remaining -= len(block)
    return digest


def save_json(path: Path, value: dict):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, indent=2)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def source_signature(path: Path):
    stat = path.stat()
    return {"path": str(path.resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def iter_chunks(path: Path, offset: int, read_size: int, max_chunk_size: int):
    """仅在完整特殊 token 之后切分；保留所有分隔符、空白和文件尾。"""
    buffer = b""
    with path.open("rb") as source:
        source.seek(offset)
        while block := source.read(read_size):
            buffer += block
            end = buffer.rfind(SEPARATOR_BYTES)
            if end >= 0:
                end += len(SEPARATOR_BYTES)
                if end > max_chunk_size:
                    raise ValueError(f"单个编码块超过 {max_chunk_size} bytes；请检查长文档并调整 --max-chunk-mib")
                yield buffer[:end]
                buffer = buffer[end:]
            if len(buffer) > max_chunk_size:
                raise ValueError(f"超过 {max_chunk_size} bytes 仍无完整文档分隔符；停止以限制内存")
        if buffer:
            yield buffer


def initialize_worker(model_path: str):
    global TOKENIZER, SPECIAL_ID
    TOKENIZER = load_tokenizer(Path(model_path))
    if not TOKENIZER.vocab or any(type(index) is not int or not 0 <= index <= 65535 for index in TOKENIZER.vocab):
        raise ValueError("词表 ID 无法全部用 uint16 保存")
    SPECIAL_ID = TOKENIZER.byts_to_id[SEPARATOR_BYTES]


def encode_chunk(raw: bytes):
    text = raw.decode("utf-8")
    ids = TOKENIZER.encode(text)
    # decode 同时检查所有 ID 属于词表；初始化时已检查整份词表的整数范围。
    if TOKENIZER.decode(ids) != text:
        raise ValueError("编码后无法还原当前输入块")
    array = np.asarray(ids, dtype=DTYPE)
    special_count = int(np.count_nonzero(array == SPECIAL_ID))
    if special_count != raw.count(SEPARATOR_BYTES):
        raise ValueError("特殊 token 数量与原始输入不一致")
    return array, special_count


def ordered_results(pool, chunks, max_pending):
    pending = deque()
    for raw in chunks:
        pending.append((raw, pool.submit(encode_chunk, raw)))
        if len(pending) >= max_pending:
            first_raw, future = pending.popleft()
            yield first_raw, future.result()
    while pending:
        raw, future = pending.popleft()
        yield raw, future.result()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-mib", type=int, default=1)
    parser.add_argument("--max-chunk-mib", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.output.suffix != ".npy":
        parser.error("--output 必须以 .npy 结尾")
    if args.workers < 1 or args.chunk_mib < 1 or args.max_chunk_mib < args.chunk_mib:
        parser.error("进程数和块大小必须为正，max-chunk-mib 不得小于 chunk-mib")
    manifest = args.output.with_suffix(".json")
    raw_path = args.output.with_name(args.output.name + ".partial.u16")
    checkpoint = args.output.with_name(args.output.name + ".progress.json")
    npy_temporary = args.output.with_name(args.output.name + ".partial")
    ready = args.output.with_name(args.output.name + ".ready.json")
    if any(path.exists() for path in (args.output, manifest, raw_path, checkpoint, npy_temporary, ready)) and not args.resume:
        parser.error("发现未完成的输出；确认继续时使用 --resume")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parent
    identity = {
        "source": source_signature(args.corpus),
        "model": {"path": str(args.model.resolve()), "sha256": hash_prefix(args.model).hexdigest()},
        "code_sha256": {
            name: hash_prefix(root / name).hexdigest()
            for name in ("encode_dataset.py", "experiment_tokenizer_a.py", "cs336_basics/tokenizer.py")
        },
        "dtype": DTYPE.str,
        "separator": SEPARATOR,
    }
    # 恢复 NPY 已验证后、最终文件/manifest 发布或清理期间发生的中断。
    if args.output.exists() or manifest.exists() or ready.exists():
        record_path = manifest if manifest.exists() else ready
        if not record_path.exists():
            raise ValueError("已有最终数组但缺少可验证的完成记录，拒绝覆盖")
        completed = json.loads(record_path.read_text())
        if completed["identity"] != identity:
            raise ValueError("已完成产物的输入、模型或代码身份不匹配")
        for partial_record in (checkpoint, ready):
            if partial_record.exists() and json.loads(partial_record.read_text())["identity"] != identity:
                raise ValueError("临时记录属于不同任务，拒绝清理")
        candidate = args.output if args.output.exists() else npy_temporary
        if hash_prefix(candidate).hexdigest() != completed["output"]["sha256"]:
            raise ValueError("待完成发布的 NPY 文件校验失败")
        if not args.output.exists():
            candidate.replace(args.output)
        if not manifest.exists():
            ready.replace(manifest)
        for temporary in (raw_path, checkpoint, npy_temporary, ready):
            if temporary.exists():
                temporary.unlink()
        print(f"已验证完成产物并清理中间文件：{args.output}", flush=True)
        return
    started = perf_counter()
    if checkpoint.exists():
        state = json.loads(checkpoint.read_text())
        if state["identity"] != identity:
            raise ValueError("输入、模型或代码与检查点不一致，拒绝继续")
    else:
        if raw_path.exists():
            raise ValueError("存在原始输出但没有检查点，不能确认恢复位置")
        state = {
            "identity": identity, "input_bytes": 0, "tokens": 0, "chunks": 0,
            "special_tokens": 0, "min_id": None, "max_id": None, "recorded_seconds": 0,
            "input_sha256": hashlib.sha256().hexdigest(), "payload_sha256": hashlib.sha256().hexdigest(),
            "runs": [],
        }
    previous_seconds = state["recorded_seconds"]
    state["runs"].append({
        "started_at_utc": datetime.now(UTC).isoformat(), "workers": args.workers,
        "chunk_mib": args.chunk_mib, "max_chunk_mib": args.max_chunk_mib,
        "resume_input_bytes": state["input_bytes"],
    })
    input_hash = hash_prefix(args.corpus, state["input_bytes"])
    if not raw_path.exists() and state["tokens"] == 0:
        raw_path.touch(exist_ok=False)
    payload_hash = hash_prefix(raw_path, state["tokens"] * DTYPE.itemsize)
    if input_hash.hexdigest() != state["input_sha256"] or payload_hash.hexdigest() != state["payload_sha256"]:
        raise ValueError("检查点对应的输入或输出前缀哈希不匹配")

    def commit(output):
        output.flush()
        os.fsync(output.fileno())
        state["input_sha256"] = input_hash.hexdigest()
        state["payload_sha256"] = payload_hash.hexdigest()
        state["recorded_seconds"] = previous_seconds + perf_counter() - started
        save_json(checkpoint, state)

    print(f"开始：{args.corpus} -> {args.output}；workers={args.workers}；恢复位置={state['input_bytes']:,} bytes", flush=True)
    last_report = perf_counter()
    chunks = iter_chunks(args.corpus, state["input_bytes"], args.chunk_mib * 1024**2, args.max_chunk_mib * 1024**2)
    with raw_path.open("r+b") as output:
        output.truncate(state["tokens"] * DTYPE.itemsize)
        output.seek(0, os.SEEK_END)
        commit(output)
        with ProcessPoolExecutor(args.workers, initializer=initialize_worker, initargs=(str(args.model),)) as pool:
            for raw, (array, special_count) in ordered_results(pool, chunks, 2 * args.workers):
                payload = array.tobytes()
                output.write(payload)
                input_hash.update(raw)
                payload_hash.update(payload)
                state["input_bytes"] += len(raw)
                state["tokens"] += array.size
                state["chunks"] += 1
                state["special_tokens"] += special_count
                if array.size:
                    state["min_id"] = min(state["min_id"], int(array.min())) if state["min_id"] is not None else int(array.min())
                    state["max_id"] = max(state["max_id"], int(array.max())) if state["max_id"] is not None else int(array.max())
                if state["chunks"] % 8 == 0:
                    commit(output)
                if perf_counter() - last_report >= 20:
                    percentage = 100 * state["input_bytes"] / identity["source"]["bytes"]
                    print(f"{percentage:.1f}%；{state['input_bytes']:,} bytes；{state['tokens']:,} tokens；本次 {perf_counter() - started:.1f}s", flush=True)
                    last_report = perf_counter()
        commit(output)
    if state["input_bytes"] != identity["source"]["bytes"] or source_signature(args.corpus) != identity["source"]:
        raise ValueError("输入未完整读取或运行期间发生改变")

    print("编码完成，写入 NPY 头并核对输出…", flush=True)
    with npy_temporary.open("wb") as destination, raw_path.open("rb") as source:
        np.lib.format.write_array_header_2_0(destination, {
            "descr": DTYPE.str, "fortran_order": False, "shape": (state["tokens"],),
        })
        shutil.copyfileobj(source, destination, length=8 * 1024**2)
        destination.flush()
        os.fsync(destination.fileno())
    saved = np.load(npy_temporary, mmap_mode="r", allow_pickle=False)
    if saved.dtype != DTYPE or saved.shape != (state["tokens"],):
        raise ValueError("NPY 的 dtype 或 shape 不符合预期")
    verified_hash = hashlib.sha256()
    for offset in range(0, saved.size, 4 * 1024**2):
        verified_hash.update(saved[offset:offset + 4 * 1024**2].tobytes())
    if verified_hash.hexdigest() != state["payload_sha256"]:
        raise ValueError("NPY 数据区与编码流不一致")
    del saved
    state["recorded_seconds"] = previous_seconds + perf_counter() - started
    state["output"] = {
        "path": str(args.output.resolve()), "shape": [state["tokens"]], "dtype": DTYPE.str,
        "file_bytes": npy_temporary.stat().st_size, "sha256": hash_prefix(npy_temporary).hexdigest(),
    }
    state["checks"] = {"roundtrip": "every input chunk", "special_token_count": "every input chunk", "npy_payload_sha256": "matched"}
    save_json(ready, state)
    npy_temporary.replace(args.output)
    ready.replace(manifest)
    raw_path.unlink()
    checkpoint.unlink()
    print(f"完成：{state['tokens']:,} tokens；ID范围 {state['min_id']}..{state['max_id']}；uint16；{state['recorded_seconds']:.1f}s", flush=True)
    print(f"数组：{args.output}；记录：{manifest}", flush=True)


if __name__ == "__main__":
    main()

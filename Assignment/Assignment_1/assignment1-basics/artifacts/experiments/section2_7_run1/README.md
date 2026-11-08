# 2.7 第一轮实验记录

本轮运行已有的 `experiment_tokenizer_a.py`，收集测量数据。书面分析尚未完成。

## 配置

- 运行目录：仓库根目录。
- Python：`.venv/bin/python`。
- 模型：`artifacts/tinystories_bpe_parallel.pkl`（10,000 项）和本次训练的 `artifacts/owt_bpe_32000.pkl`（32,000 项）。
- 抽样：每份训练语料 10 篇文档，seed=42，使用现有脚本的 reservoir sampling。
- 文档按 `<|endoftext|>` 分隔，分隔符不计入本次样本文本；忽略仅含空白的文档。
- 字节数按 UTF-8 统计；日志的总体比率为总字节数 / 总 token 数。

## 实际运行的实验命令

```sh
.venv/bin/python -u experiment_tokenizer_a.py --dataset TinyStories --corpus data/TinyStoriesV2-GPT4-train.txt --model artifacts/tinystories_bpe_parallel.pkl --samples-out artifacts/experiments/section2_7_run1/tiny_samples.jsonl --seed 42 --num-docs 10
.venv/bin/python -u experiment_tokenizer_a.py --dataset OpenWebText_with_TinyStories_tokenizer --corpus data/owt_train.txt --model artifacts/tinystories_bpe_parallel.pkl --samples-out artifacts/experiments/section2_7_run1/owt_samples.jsonl --seed 42 --num-docs 10
.venv/bin/python -u experiment_tokenizer_a.py --dataset OpenWebText --samples-in artifacts/experiments/section2_7_run1/owt_samples.jsonl --model artifacts/owt_bpe_32000.pkl --samples-out artifacts/experiments/section2_7_run1/owt_samples_owt_tokenizer.jsonl --num-docs 10
```

标准输出分别保存至 `a_tinystories.log`、`b_tinystories_on_owt.log` 和 `a_openwebtext.log`，三次进程退出码均为 0。第三次运行直接读取已保存的 OWT 样本，没有重新抽样。

## 观测结果

| 语料 | Tokenizer | 语料文档数（初次扫描） | 抽样文档数 | 样本字节数 | Token 数 | bytes/token |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| TinyStories train | TinyStories 10K | 2,172,281 | 10 | 6,734 | 1,629 | 4.1338 |
| OpenWebText train | TinyStories 10K | 2,399,397 | 10 | 88,949 | 26,603 | 3.3436 |
| OpenWebText train | OpenWebText 32K | 2,399,397 | 10 | 88,949 | 19,611 | 4.5357 |

三组测量均通过 `check_tokenizer_inputs.py` 的文档数量、格式和 encode/decode 往返检查。`cmp` 确认 `owt_samples_owt_tokenizer.jsonl` 与原有 `owt_samples.jsonl` 逐字节一致，两次 OWT 测量使用相同文档、顺序及正文。
本轮 TinyStories 样本与原有 `artifacts/experiments/tiny_samples.jsonl` 的文档编号不同；原文件保留，本轮结果只对应本目录的样本。

## 完成状态

- (a)：TinyStories + TinyStories tokenizer、OpenWebText + OpenWebText tokenizer 两组测量均已完成；书面回答待自行整理。
- (b)：同一批 OWT 文档的两种 tokenizer 测量均已完成，可对照表中后两行；书面分析待自行完成。
- (c)：尚未测量编码吞吐量或估算 Pile 耗时。抽样扫描包括读取整个语料，不可直接当成编码吞吐量。
- (d)：尚未生成四份全量训练／验证 token ID 序列。

后续对比仍需要本目录的样本与日志，因此保留这些文件。

## OpenWebText 32K 训练

训练入口 `train_bpe_model.py` 调用现有 `cs336_basics/bpe.py`，仅添加参数、阶段进度及模型保存。预分词使用现有函数支持的分块数和进程数参数；未修改 BPE 算法源码。

```sh
.venv/bin/python -u train_bpe_model.py --corpus data/owt_train.txt --output artifacts/owt_bpe_32000.pkl --vocab-size 32000 --workers 4 --pretoken-chunks 128 --max-memory-gib 10
```

- 输入：完整的 `data/owt_train.txt`，11,920,511,059 bytes。
- 特殊 token：`<|endoftext|>`；目标词表 32,000。
- 日志：`train_owt_32000.log`。内存限制为每进程 10 GiB 虚拟地址空间，并非整组进程的 RSS 上限。
- 状态：训练和本组编码测量均已完成。模型包含 32,000 个词表项、31,743 次合并，保存为 `artifacts/owt_bpe_32000.pkl`（748,109 bytes）。
- 本次完成训练的运行耗时：5,789.46 秒（约 96.49 分钟），不含之前停止的单进程尝试。
- 阶段耗时：预分词 327.99 秒，字节序列转换 7.34 秒，pair 统计初始化 24.02 秒；其余主要为合并阶段。
- 主进程峰值 RSS：8,062.57 MiB（约 7.87 GiB）；这是主进程指标，不是所有进程峰值之和。
- 模型中的 `metadata` 记录了语料绝对路径、字节数、修改时间、训练参数、耗时及 BPE 源文件哈希。
- 最初的 16 块 / 1 进程尝试在预分词阶段主动停止，未生成模型；日志保留为 `train_owt_32000_single_worker.log`。
- 配置回归检查：在 `tests/fixtures/tinystories_sample_5M.txt` 上，16 块 / 1 进程与 128 块 / 4 进程训练到 1,000 词表，vocab 和 merges 完全一致。
- 现有 BPE 测试：`tests/test_train_bpe.py` 的 3 项测试全部通过。
- `experiment_tokenizer_a.py` 已增加 `--samples-in`，用于复用本目录原有 OWT 样本。对 TinyStories 模型的回归测量仍为 88,949 bytes / 26,603 tokens，输出样本文本与原样本逐字节一致。

## 文件版本（SHA-256）

最初两组测量使用的版本：

```text
8f1fe0342c5761561b93079333b11ff7602508b440ea73d95cda718e29bc04cf  artifacts/tinystories_bpe_parallel.pkl
fcaf81eb2a14ed514288618d8a0a415bbd0f58b262454bbf0ec97bbabfb4b5a7  experiment_tokenizer_a.py
290d0d15de0198a78613fb4ed6d3a137aaf6c4f557def8dfcae110371ef9b393  cs336_basics/tokenizer.py
```

补齐 OpenWebText 组时的代码和产物：

```text
64a686ae1e54fcc8c82979de63a081aba2d03b4fa01d5d2410ddc125047e2726  artifacts/owt_bpe_32000.pkl
13c67101f661cb0c2a5dee60c0b00aa5f2089f3f6831d6f6155694952859e2a5  artifacts/experiments/section2_7_run1/owt_samples.jsonl
b5028cf10eb4da5a3b94f168b9a69f45996ce98369e0890220a42b887647ed4b  train_bpe_model.py
87aa746aca55e0ecde2943ca7d92d8ca47cb1c38721572b7e20b735588c15855  experiment_tokenizer_a.py
d3895672fb7b5f85ad33681c560e87205f9145a9b69e65128ae33bc3a8dd63d7  cs336_basics/bpe.py
290d0d15de0198a78613fb4ed6d3a137aaf6c4f557def8dfcae110371ef9b393  cs336_basics/tokenizer.py
```

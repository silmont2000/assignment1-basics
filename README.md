# CS336 Spring 2025 Assignment 1: Basics

这里是我个人对作业1的实现，仅供参考。约90%手搓，10% vibe coding

刚刚入门，如有错误或者问题，随时欢迎提isu讨论，不胜感激！

如果需要我的分词表或.bin数据，可以发邮件 xbyzju@gmail.com

如果对你有用，欢迎点点star ~

## 目录结构

- `bpe/`：BPE 分词相关实现

  - `bpe_tokenizer.py`：BPE Tokenizer 定义
  - `bpe_trainer.py`：训练 BPE 词表的逻辑
  - `common.py`：BPE 部分的公共工具函数
  - `parallel_executor.py`：并行执行/统计相关代码
- `model/`：模型模块

  - `embedding.py`：词嵌入与位置嵌入
  - `linear.py`：线性层实现
  - `multihead_attention.py`：多头自注意力模块
  - `rmsnorm.py`：RMSNorm 层
  - `rope.py`：RoPE 旋转位置编码
  - `swiglu.py`：SwiGLU 激活模块
  - `transformer_block.py`：Transformer Block 组合
  - `common.py`：模型相关函数
- `train/`：训练相关工具

  - `adamW.py`：AdamW 优化器实现
  - `ckpt.py`：模型 checkpoint 保存/加载
  - `get_batch.py`：从数据中构造训练 batch
  - `common.py`：训练相关公共配置和函数
- 顶层脚本

  - `pretokenization.py`：预分词/数据预处理脚本
  - `pretokenization_example.py`：预分词使用示例（原来就带着的我就没删）
  - `transformer_lm.py`：Transformer 语言模型封装
  - `train_loop.py`：训练循环逻辑
  - `run.py`：入口脚本，运行 demo

## 说明

1. 准备数据并进行预分词

   - 阅读 `pretokenization.py` 和 `pretokenization_example.py`
   - 将原始文本转为适合 BPE/模型训练的格式
2. 训练 BPE 分词器

   - 查看 `bpe/bpe_trainer.py`，根据代码注释指定训练语料和超参数
   - 生成 BPE 词表和分词模型文件
3. 构建并训练 Transformer 语言模型

   - `transformer_lm.py` 中定义了完整模型结构
   - `train_loop.py` 中实现了训练循环
   - `train/` 目录提供优化器、batch 构造、checkpoint 等工具
   - 结合 `run.py` 中的示例代码运行训练过程

## Setup

### Environment

We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using

```sh
uv run <python_file_path>
```

and the environment will be automatically solved and activated when necessary.

### Run unit tests

```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Download data

Download the TinyStories data and a subsample of OpenWebText

```sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

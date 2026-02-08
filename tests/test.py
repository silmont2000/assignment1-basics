import sys
import os

from cs336_basics.bpe.parallel_executor import bpe_trainer, parallel_word_counts
from cs336_basics.pretokenization_example import find_chunk_boundaries


# input_path = 'assignment1-basics/data/small_story.txt'
# f = open(input_path, "rb")
# statistics = global_initialization(f.read().decode())

# a_tokenizer = tokenizer(
#     byte_string=f.read(),
#     vocab_size=500,
#     special_tokens=["<|endoftext|>"],
#     statistics=statistics)
# a_tokenizer.run()

# f.close()


if __name__ == "__main__":
    # --- 1. 配置参数 ---
    input_path = 'assignment1-basics/data/small_story.txt'  # 你的语料库路径
    num_proc = 4                              # 开启进程数
    target_vocab_size = 300                   # 目标词表大小

    print(f"正在并行统计单词频次 (进程数: {num_proc})...")
    word_counts_dict = parallel_word_counts(
        input_path, num_processes=num_proc,  special_tokens=["<|endoftext|>"])
    print(f"统计完成，共有 {len(word_counts_dict)} 个独特单词。")

    print("正在初始化 BPE 训练器...")
    bpe_trainer = bpe_trainer(
        vocab_size=target_vocab_size,
        special_tokens=["<|endoftext|>"],
        pat_string=word_counts_dict
    )

    print(f"开始训练，目标词表大小: {target_vocab_size}...")
    bpe_trainer.run()

    print("训练完成！")
    print(f"最终合并规则数量: {len(bpe_trainer.merges)}")

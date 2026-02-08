import sys
import os

from cs336_basics.bpe.bpe_trainer import bpe_trainer
from cs336_basics.bpe.parallel_executor import parallel_word_counts
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

input_path = 'assignment1-basics/data/small_story.txt'  # 你的语料库路径
num_proc = 6                              # 开启进程数
target_vocab_size = 32000                   # 目标词表大小

if __name__ == "__main__":
    import time

    # --- 1. 配置参数 ---

    total_start_time = time.time()

    print(f"正在并行统计单词频次 (进程数: {num_proc})...")
    t0 = time.time()
    word_counts_dict = parallel_word_counts(
        input_path, num_processes=num_proc,  special_tokens=["<|endoftext|>"])
    t1 = time.time()
    count_time = t1 - t0
    print(f"统计完成，共有 {len(word_counts_dict)} 个独特单词。耗时: {count_time:.2f} 秒")

    print("正在初始化 BPE 训练器...")
    t2 = time.time()
    bpe_trainer = bpe_trainer(
        vocab_size=target_vocab_size,
        special_tokens=["<|endoftext|>"],
        pat_string=word_counts_dict,
        # load_from_file=True
    )
    t3 = time.time()
    init_time = t3 - t2
    print(f"初始化完成。耗时: {init_time:.2f} 秒")

    print(f"开始训练，目标词表大小: {target_vocab_size}")
    t4 = time.time()
    bpe_trainer.run(True)
    t5 = time.time()
    train_time = t5 - t4
    print(f"训练完成！耗时: {train_time:.2f} 秒")
    print(f"最终合并规则数量: {len(bpe_trainer.merges)}")

    total_time = time.time() - total_start_time

    print("\n" + "="*30)
    print("       耗时统计报告")
    print("="*30)
    print(f"1. 并行统计词频: {count_time:.2f} s")
    print(f"2. BPE 初始化  : {init_time:.2f} s")
    print(f"3. BPE 训练    : {train_time:.2f} s")
    print("-" * 30)
    print(f"总耗时         : {total_time:.2f} s")
    print("="*30 + "\n")

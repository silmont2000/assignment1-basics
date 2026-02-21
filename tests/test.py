# import sys
# import os

# from cs336_basics.bpe.bpe_tokenizer import bpe_tokenizer
# from cs336_basics.bpe.bpe_trainer import bpe_trainer
# from cs336_basics.bpe.parallel_executor import parallel_word_counts
# from cs336_basics.pretokenization_example import find_chunk_boundaries
# from cs336_basics.bpe.common import *

# # input_path = 'assignment1-basics/data/TinyStoriesV2-GPT4-train.txt'  # 你的语料库路径
# # input_path = 'assignment1-basics/data/owt_train.txt'  # 你的语料库路径
# input_path = 'assignment1-basics/data/small_story.txt'  # 你的语料库路径
# num_proc = 8                              # 开启进程数
# target_vocab_size = 32000                   # 目标词表大小


# 这里是跑分词表的代码
# 这里是跑分词表的代码
# 这里是跑分词表的代码

# if __name__ == "__main__":
#     import time

#     # --- 1. 配置参数 ---

#     total_start_time = time.time()

#     print(f"正在并行统计单词频次 (进程数: {num_proc})...")
#     t0 = time.time()
#     word_counts_dict = parallel_word_counts(
#         input_path, num_processes=num_proc,  special_tokens=["<|endoftext|>"])
#     t1 = time.time()
#     count_time = t1 - t0
#     print(f"统计完成，共有 {len(word_counts_dict)} 个独特单词。耗时: {count_time:.2f} 秒")

#     print("正在初始化 BPE 训练器...")
#     t2 = time.time()
#     bpe_trainer = bpe_trainer(
#         vocab_size=target_vocab_size,
#         special_tokens=["<|endoftext|>"],
#         pat_string=word_counts_dict,
#         load_from_file=True,
#         # file_path='tokenizer_owt_train.pkl'
#     )
#     t3 = time.time()
#     init_time = t3 - t2
#     print(f"初始化完成。耗时: {init_time:.2f} 秒")

#     print(f"开始训练，目标词表大小: {target_vocab_size}")
#     t4 = time.time()
#     bpe_trainer.run(True)
#     t5 = time.time()
#     train_time = t5 - t4
#     print(f"训练完成！耗时: {train_time:.2f} 秒")
#     print(f"最终合并规则数量: {len(bpe_trainer.merges)}")

#     total_time = time.time() - total_start_time

#     print("\n" + "="*30)
#     print("       耗时统计报告")
#     print("="*30)
#     print(f"1. 并行统计词频: {count_time:.2f} s")
#     print(f"2. BPE 初始化  : {init_time:.2f} s")
#     print(f"3. BPE 训练    : {train_time:.2f} s")
#     print("-" * 30)
#     print(f"总耗时         : {total_time:.2f} s")
#     print("="*30 + "\n")

# with open(input_path, "r") as f:
#     data = load_trained_bpe()
#     tokenizer = bpe_tokenizer(
#         vocab=data['vocab'], merges=data['merges'], special_tokens=['<|endoftext|>'])
#     ids = tokenizer.encode(f.read())
#     print(ids)
#     print(tokenizer.decode(ids))

# print("\n==============================\n       耗时统计报告\n==============================\n1. 并行统计词频: 129.23 s\n2. BPE 初始化  : 0.00 s\n3. BPE 训练    : 9.67 s\n------------------------------\n总耗时         : 138.90 s\n==============================")

import torch
import time

device = torch.device("mps")
size = 8192
x = torch.randn(size, size, device=device)
y = torch.randn(size, size, device=device)

print("Starting heavy GPU load...")
start_time = time.time()

for i in range(500):
    z = torch.mm(x, y)
    z = torch.nn.functional.relu(z)

    if i % 50 == 0:
        # synchronize 是关键：让 CPU 等待 GPU 完成
        torch.mps.synchronize()
        print(f"Iteration {i}, last value: {z[0,0].item():.4f}")

print(f"Total time: {time.time() - start_time:.2f}s")

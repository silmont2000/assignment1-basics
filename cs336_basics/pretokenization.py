import numpy as np
import concurrent.futures
from functools import partial
import time
import os
from tqdm import tqdm

from cs336_basics.pretokenization_example import find_chunk_boundaries
from cs336_basics.bpe.common import load_trained_bpe
from cs336_basics.bpe.bpe_tokenizer import bpe_tokenizer


def process_chunk_to_disk(start_end_idx, file_path, tokenizer, temp_dir):
    (start, end), idx = start_end_idx
    with open(file_path, "rb") as f:
        f.seek(start)
        # 记录原始读取的字节数，用于进度条统计
        raw_bytes = f.read(end - start)
        chunk_text = raw_bytes.decode("utf-8", errors="ignore")

    # 直接调用 encode，内部处理效率更高
    ids = tokenizer.encode(chunk_text)

    temp_file = os.path.join(temp_dir, f"chunk_{idx:05d}.npy")
    np.save(temp_file, np.array(ids, dtype=np.uint16))
    # 返回 token 数和处理的字节数
    return len(ids), len(raw_bytes)


def parallel_tokenize_extreme(input_file, tokenizer, target_path, num_processes=12):
    start_time = time.time()
    file_size = os.path.getsize(input_file)
    temp_dir = "temp_tokens"
    os.makedirs(temp_dir, exist_ok=True)

    # --- 修改 1: 细化切片，约每 10MB 一个进度单位 ---
    num_chunks = max(num_processes * 4, file_size // (10 * 1024 * 1024))
    print(f"[{time.strftime('%H:%M:%S')}] 阶段 1: 扫描边界，计划切分 {num_chunks} 块...")

    with open(input_file, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, int(num_chunks), b"<|endoftext|>")

    chunk_pairs = list(zip(boundaries[:-1], boundaries[1:]))
    indexed_chunks = list(zip(chunk_pairs, range(len(chunk_pairs))))

    # 2. 多进程编码
    print(f"[{time.strftime('%H:%M:%S')}] 阶段 2: 编码中...")
    worker_func = partial(process_chunk_to_disk, file_path=input_file,
                          tokenizer=tokenizer, temp_dir=temp_dir)

    token_counts = [0] * len(indexed_chunks)

    # --- 修改 2: 在 tqdm 中增加 unit='B' 和 unit_scale，实时看 MB/s ---
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = {executor.submit(worker_func, item)
                                   : item[1] for item in indexed_chunks}

        with tqdm(total=file_size, desc="Encoding Data", unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                n_tokens, n_bytes = future.result()
                token_counts[idx] = n_tokens
                pbar.update(n_bytes)  # 进度条按处理的字节数滑动

    # 3. 最终合并
    print(f"[{time.strftime('%H:%M:%S')}] 阶段 3: 合并文件中...")
    total_tokens = sum(token_counts)
    final_mmap = np.memmap(target_path, dtype=np.uint16,
                           mode='w+', shape=(total_tokens,))

    current_pos = 0
    for i in tqdm(range(len(indexed_chunks)), desc="Merging"):
        temp_file = os.path.join(temp_dir, f"chunk_{i:05d}.npy")
        chunk_data = np.load(temp_file, mmap_mode="r")
        count = len(chunk_data)
        final_mmap[current_pos: current_pos + count] = chunk_data
        current_pos += count
        os.remove(temp_file)

    final_mmap.flush()
    os.rmdir(temp_dir)

    end_time = time.time()
    print(f"\n{'='*40}")
    print(f"处理完成！")
    print(f"总 Token 数: {total_tokens:,}")
    print(f"总计耗时: {end_time - start_time:.2f} 秒")
    print(f"平均速度: {total_tokens/(end_time - start_time):.2f} tokens/s")
    print(f"{'='*40}")


if __name__ == "__main__":
    # 配置
    pkl_path = 'assignment1-basics/tokenizer/tokenizer_tinystory.pkl'
    input_data = 'assignment1-basics/data/TinyStoriesV2-GPT4-train.txt'
    output_bin = 'TinyStoriesV2-GPT4-train-token.bin'

    data = load_trained_bpe(pkl_path)
    tokenizer = bpe_tokenizer(
        vocab=data["vocab"],
        merges=data["merges"],
        special_tokens=["<|endoftext|>"]
    )

    # 既然 CPU 还有富裕，尝试设置到 12-16 (视你的核心数而定)
    parallel_tokenize_extreme(input_data, tokenizer,
                              output_bin, num_processes=12)

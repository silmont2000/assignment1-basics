from cs336_basics.pretokenization_example import find_chunk_boundaries
from cs336_basics.bpe.bpe_trainer import bpe_trainer
import regex as re
from concurrent.futures import ProcessPoolExecutor
from collections import Counter


def merge_statistics(stats1, stats2):
    for word, info in stats2.items():
        if word in stats1:
            stats1[word]['times'] += info['times']
        else:
            stats1[word] = info
    return stats1


def _process_chunk(file_path: str, start: int, end: int) -> dict:
    with open(file_path, "rb") as f:
        f.seek(start)
        chunk_bytes = f.read(end - start)
        text = chunk_bytes.decode("utf-8", errors="ignore")

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    tokens = re.findall(PAT, text)

    return Counter(tokens)


def parallel_word_counts(input_path: str, num_processes: int = 4) -> dict:
    """Parallel processing to get initial word counts from a large file."""
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    tasks = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        tasks.append((input_path, start, end))

    total_counts = Counter()
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [executor.submit(_process_chunk, *task) for task in tasks]

        for future in futures:
            chunk_counts = future.result()
            total_counts.update(chunk_counts)

    return dict(total_counts)

import codecs
from cs336_basics.pretokenization_example import find_chunk_boundaries
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


def _process_chunk(file_path: str, start: int, end: int, special_tokens: list[str]) -> dict:
    PAT = re.compile(
        """'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    special_pat = re.compile("|".join(re.escape(st)
                             for st in special_tokens)) if special_tokens else None

    chunk_counts = Counter()

    decoder = codecs.getincrementaldecoder("utf-8")(errors='replace')

    with open(file_path, "rb") as f:
        f.seek(start)
        pos = start
        buffer = ""
        # 1MB buffer size
        CHUNK_SIZE = 8 * 1024 * 1024

        while pos < end:
            read_len = min(CHUNK_SIZE, end - pos)
            b = f.read(read_len)
            if not b:
                break

            # Decode bytes to string
            text_chunk = decoder.decode(b, final=False)
            buffer += text_chunk
            pos += len(b)

            # Determine how much of the buffer we can process safely
            # We look for the last whitespace to avoid splitting tokens
            last_safe_index = -1
            # If we are at the end of the assigned chunk, we process everything
            if pos >= end:
                last_safe_index = len(buffer)
            else:
                # Find last whitespace from the end
                for i in range(len(buffer) - 1, -1, -1):
                    if buffer[i].isspace():
                        last_safe_index = i + 1  # Include the whitespace
                        break

            # If no whitespace found and buffer is small, continue accumulating
            if last_safe_index == -1 and len(buffer) < 10 * CHUNK_SIZE:
                continue

            # If buffer is huge and no whitespace, we must force split (unlikely but safe fallback)
            if last_safe_index == -1:
                last_safe_index = len(buffer)

            to_process = buffer[:last_safe_index]
            buffer = buffer[last_safe_index:]

            # Process the safe chunk
            if special_pat:
                fragments = special_pat.split(to_process)
            else:
                fragments = [to_process]

            for fragment in fragments:
                if not fragment:
                    continue
                tokens = re.findall(PAT, fragment)
                for t in tokens:
                    byte_tuple = tuple(bytes([b]) for b in t.encode("utf-8"))
                    chunk_counts[byte_tuple] += 1

    return chunk_counts


def parallel_word_counts(input_path: str, num_processes: int, special_tokens: list[str]) -> dict:
    """Parallel processing to get initial word counts from a large file."""
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    tasks = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        tasks.append((input_path, start, end, special_tokens))

    total_counts = Counter()
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [executor.submit(_process_chunk, *task) for task in tasks]

        for future in futures:
            chunk_counts = future.result()
            total_counts.update(chunk_counts)

    return dict(total_counts)

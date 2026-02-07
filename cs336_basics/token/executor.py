from cs336_basics.pretokenization_example import find_chunk_boundaries
import os
import regex as re
import pprint
from collections import Counter
# # Usage
# with open('assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt', "rb") as f:
#     num_processes = 4
#     boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

#     # The following is a serial implementation, but you can parallelize this
#     # by sending each start/end pair to a set of processes.
#     for start, end in zip(boundaries[:-1], boundaries[1:]):
#         f.seek(start)
#         chunk = f.read(end - start).decode("utf-8", errors="ignore")
#         # Run pre-tokenization on your chunk and store the counts for each pre-token

input_file = 'assignment1-basics/data/small_story.txt'
times = 'times'
pairs = 'pairs'
tokens = 'tokens'


class tokenizer:
    def __init__(self,    text_string: str,
                 vocab_size: int,
                 special_tokens: list[str]):
        self.text_string = text_string
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens

    def append_merge(self, token):
        self.merges.append(token)

    def append_vocab(self, token: bytes):
        self.vocab[len(self.vocab)+1] = token

    def pre_initialization(self, text):
        # vocab: dict[int, bytes] 分词器词汇表，从 int（词汇表中的标记 ID）到 bytes（标记字节）的映射。
        # merges: list[tuple[bytes, bytes]] 训练过程中生成的 BPE 合并列表。每个列表项
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges: list[tuple[bytes, bytes]] = []
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        pat_string = re.findall(PAT, text)

        counts = {}
        for item in pat_string:
            counts[item] = counts.get(item, 0) + 1
        # print(counts)

        self.statistics = {}
        for k, v in counts.items():
            if len(k) < 2:
                self.statistics[k] = {
                    times: v,
                    pairs: Counter(),
                    tokens: [i.encode("utf-8") for i in k]
                }
                continue

            self.statistics[k] = {
                times: v,
                pairs: Counter(),
                tokens: [i.encode("utf-8") for i in k]

            }
            for i in range(len(k)-1):
                cur_pair = (k[i].encode(), k[i+1].encode())
                self.statistics[k][pairs][cur_pair] += 1

    def update_pairs_in_words(self, max_pair: tuple):
        new_token = b''.join(max_pair)
        for _, (k, v) in enumerate(self.statistics.items()):
            # 先删去max_pair
            cur_pairs = v[pairs]
            cur_tokens = v[tokens]
            result_tokens = []
            del cur_pairs[max_pair]
            i = 0
            while i < len(cur_tokens)-1:
                if (cur_tokens[i], cur_tokens[i+1]) == max_pair:
                    if i != 0:
                        left_new_pair = (cur_tokens[i-1], max_pair)
                        left_old_pair = (cur_tokens[i-1], cur_tokens[i])
                        cur_pairs[left_old_pair] -= 1
                        cur_pairs[left_new_pair] += 1

                    if i != len(cur_tokens)-1:
                        right_new_pair = (new_token, cur_tokens[i+2])
                        right_old_pair = (cur_tokens[i+1], cur_tokens[i+2])
                        cur_pairs[right_old_pair] -= 1
                        cur_pairs[right_new_pair] += 1

                    result_tokens.append(new_token)
                    i += 1
                else:
                    result_tokens.append(cur_tokens[i])
                i += 1

            result_tokens.append(cur_tokens[i])

            v[tokens] = result_tokens
            cur_pairs = +cur_pairs

    def find_max_pair(self):
        tmp = Counter()
        for _, (_, v) in enumerate(self.statistics.items()):
            for pair in v[pairs]:
                tmp[pair] += v[pairs][pair] + v[times]
        return tmp.most_common(1)

    def run(self):
        self.pre_initialization(self.text_string)

        max_pair = self.find_max_pair()
        print(max_pair)
        self.print_statistics()

        self.append_merge(max_pair[0][0])
        self.update_pairs_in_words(max_pair[0][0])
        self.print_statistics()

    def print_statistics(self):
        print("\n\n======Displaying first 5 statistics items======")
        for i, (k, v) in enumerate(self.statistics.items()):
            if i >= 5:
                break
            print(f"Token: {repr(k)}")
            pprint.pprint(v)
            print("-" * 20)

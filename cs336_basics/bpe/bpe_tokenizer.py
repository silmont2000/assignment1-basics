from collections import defaultdict
from cs336_basics.bpe.common import *
from cs336_basics.bpe.parallel_executor import parallel_word_counts
from collections import Counter
import regex as re
from typing import Iterable, Iterator

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class bpe_tokenizer:
    def __init__(self,
                 vocab: dict[int, bytes],
                 merges: list[tuple[bytes, bytes]],
                 special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.reverse_vocab = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.special_tokens: list[str] = [] if (
            special_tokens is None) else special_tokens

        # self.statistics, self.pair_to_statistics = statistics_initialization(
        #     self.pat_string)

    # @classmethod
    # def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):

    # def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:

    def update_pairs_in_words(self, max_pair: tuple, pair_to_statistics: defaultdict, statistics: dict):
        new_token = b''.join(max_pair)
        affected_words = pair_to_statistics[max_pair]

        for k in affected_words:
            v = statistics[k]
            cur_tokens = v[tokens]
            result_tokens = []
            i = 0
            while i < len(cur_tokens):
                if (i < len(cur_tokens)-1) and ((cur_tokens[i], cur_tokens[i+1]) == max_pair):
                    if i > 0:
                        left_new_pair = (result_tokens[-1], new_token)
                        pair_to_statistics[left_new_pair].add(k)

                    if i < len(cur_tokens)-2:
                        right_new_pair = (new_token, cur_tokens[i+2])
                        pair_to_statistics[right_new_pair].add(k)

                    i += 2
                    result_tokens.append(new_token)
                else:
                    result_tokens.append(cur_tokens[i])
                    i += 1

            v[tokens] = result_tokens

    def normal_encode(self, text: str) -> list[int]:
        words_to_token__cache: dict[bytes, list[int]] = {}
        # for i in self.special_tokens:
        #     token = i.encode()
        #     words_to_token__cache[token] = [self.reverse_vocab[token]]

        patted_words = re.findall(PAT, text)
        bytes_words = []
        for t in patted_words:
            byte_tuple = tuple(bytes([b]) for b in t.encode("utf-8"))
            bytes_words.append(byte_tuple)
        statistics, pair_to_statistics, _ = statistics_initialization(
            Counter(bytes_words))

        # 这里先按merges完成分词
        for rule in self.merges:
            self.update_pairs_in_words(
                rule, statistics=statistics, pair_to_statistics=pair_to_statistics)

        output = []
        for word in bytes_words:
            cache = words_to_token__cache.get(word)
            if cache:
                output.extend(cache)
            else:
                cache = []
                statistic = statistics[word]
                token_list = statistic[tokens]
                for token in token_list:
                    cache.append(self.reverse_vocab[token])
                words_to_token__cache[word] = cache
                output.extend(cache)
            # print("encode ", word, " to ", cache)

        return output

    def encode(self, text: str) -> list[int]:
        if not self.special_tokens:
            return self.normal_encode(text)

        sorted_special = sorted(self.special_tokens, key=len, reverse=True)

        special_pattern = re.compile(
            "(" + "|".join(re.escape(k) for k in sorted_special) + ")")
        parts = special_pattern.split(text)

        final_ids = []
        for part in parts:
            if part in self.special_tokens:
                # 如果是特殊 Token，直接给 ID
                token = part.encode()
                final_ids.append(self.reverse_vocab[token])
                # print("encode ", part, " to ", [self.reverse_vocab[token]])
            elif part:
                # 如果是普通文本，走你之前的 BPE 逻辑（包含预分词正则、Cache 等）
                final_ids.extend(self.normal_encode(part))

        return final_ids

    def decode(self, ids: list[int]) -> str:
        output: list[bytes] = []
        for i in ids:
            output.append(self.vocab[i])

        return b''.join(output).decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        输入是一个可迭代对象，每次给出一个字符串。
        输出是一个迭代器，按顺序产出所有的 token ID。
        """
        for text in iterable:
            token_ids = self.encode(text)
            yield from token_ids

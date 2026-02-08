import pprint
from collections import Counter
from collections import defaultdict
import heapq
import pickle

times = 'times'
pairs = 'pairs'
tokens = 'tokens'


class bpe_trainer:
    def __init__(self,
                 vocab_size: int,
                 special_tokens: list[str],
                 pat_string: dict,
                 load_from_file: bool = False,
                 file_path='tokenizer.pkl'):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.special_tokens_len = len(special_tokens)
        self.pat_string = pat_string
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges: list[tuple[bytes, bytes]] = []
        if load_from_file:
            self.load_trained_bpe(file_path)

        self.still_exist_multi_token = True
        self.pair_counter = Counter()

    def append_merge(self, token):
        self.merges.append(token)

    def append_vocab(self, token: bytes):
        self.vocab[len(self.vocab)] = token

    def global_initialization(self):
        self.statistics = {}
        self.pair_to_statistics = defaultdict(set)
        for k, v in self.pat_string.items():
            if len(k) < 2:
                self.statistics[k] = {
                    times: v,
                    pairs: Counter(),
                    tokens: [i for i in k]
                }
                continue

            self.statistics[k] = {
                times: v,
                pairs: Counter(),
                tokens: [i for i in k]

            }
            for i in range(len(k)-1):
                cur_pair = (k[i], k[i+1])
                self.statistics[k][pairs][cur_pair] += 1
                self.pair_to_statistics[cur_pair].add(k)

        # 初始化一下全局counter
        self.find_max_pair()
        return self.statistics

    def update_pairs_in_words(self, max_pair: tuple):
        new_token = b''.join(max_pair)
        affected_words = self.pair_to_statistics[max_pair]

        for k in affected_words:
            v = self.statistics[k]
            cur_weight = v[times]
            cur_pairs = v[pairs]
            cur_tokens = v[tokens]
            result_tokens = []
            i = 0
            while i < len(cur_tokens):
                if (i < len(cur_tokens)-1) and ((cur_tokens[i], cur_tokens[i+1]) == max_pair):
                    if i > 0:
                        left_new_pair = (result_tokens[-1], new_token)
                        left_old_pair = (result_tokens[-1], cur_tokens[i])
                        cur_pairs[left_old_pair] -= 1
                        cur_pairs[left_new_pair] += 1
                        self.pair_to_statistics[left_new_pair].add(k)
                        self.pair_counter[left_old_pair] -= cur_weight
                        self.pair_counter[left_new_pair] += cur_weight
                        heapq.heappush(
                            self.maxheap, (-self.pair_counter[left_old_pair], left_old_pair))
                        heapq.heappush(
                            self.maxheap, (-self.pair_counter[left_new_pair], left_new_pair))

                    if i < len(cur_tokens)-2:
                        right_new_pair = (new_token, cur_tokens[i+2])
                        right_old_pair = (cur_tokens[i+1], cur_tokens[i+2])
                        cur_pairs[right_old_pair] -= 1
                        cur_pairs[right_new_pair] += 1
                        self.pair_to_statistics[right_new_pair].add(k)
                        self.pair_counter[right_old_pair] -= cur_weight
                        self.pair_counter[right_new_pair] += cur_weight
                        heapq.heappush(
                            self.maxheap, (-self.pair_counter[right_old_pair], right_old_pair))
                        heapq.heappush(
                            self.maxheap, (-self.pair_counter[right_new_pair], right_new_pair))

                    i += 2
                    result_tokens.append(new_token)
                else:
                    result_tokens.append(cur_tokens[i])
                    i += 1

            v[tokens] = result_tokens
            del cur_pairs[max_pair]
            self.pair_counter[max_pair] = 0
            v[pairs] = +cur_pairs

        del self.pair_to_statistics[max_pair]

    def find_max_pair(self):
        best_pairs = []
        actual_max = -1
        heap = False
        if len(self.pair_counter) == 0:
            # if len(self.pair_counter) != -1:
            tmp = Counter()
            for _, (_, v) in enumerate(self.statistics.items()):
                for pair in v[pairs]:
                    tmp[pair] += v[pairs][pair] * v[times]
            actual_max = max(tmp.values())
            self.pair_counter = tmp
            # 初始化堆 (负号是为了实现大顶堆)
            self.maxheap = [(-count, pair)
                            for pair, count in self.pair_counter.items()]
            heapq.heapify(self.maxheap)

            best_pairs = [p for p, f in tmp.items() if f == actual_max]
        else:
            heap = True
            actual_max = -1
            while self.maxheap:
                neg_freq, pair = heapq.heappop(self.maxheap)
                cur_freq = -neg_freq
                # 这个不一定是真的最大值，和当前值比较一下，如果是旧的，因为已经pop了也就删了，所以不太影响
                actual_freq = self.pair_counter.get(pair, 0)
                if actual_max < 0 and cur_freq == actual_freq and actual_freq > 0:
                    actual_max = actual_freq
                    best_pairs.append(pair)
                elif cur_freq == actual_max and cur_freq == actual_freq and actual_freq > 0:
                    best_pairs.append(pair)
                elif cur_freq < actual_max:
                    heapq.heappush(self.maxheap, (neg_freq, pair))
                    break

        best_pairs.sort(reverse=True)
        i = 1
        while heap and i < len(best_pairs):
            heapq.heappush(self.maxheap, (-actual_max, best_pairs[i]))
            i += 1

        return [(best_pairs[0], actual_max)]

    def run(self, save=False):
        self.still_exist_multi_token = False
        print(">>> 正在进行全局统计初始化...")
        self.global_initialization()
        print(">>> 全局统计初始化完成。")

        for token in self.special_tokens:
            self.append_vocab(token.encode())
        print(f">>> 特殊 Token 已添加到词表。当前词表大小: {len(self.vocab)}")

        print(">>> 开始 BPE 合并循环...")
        while True:
            max_pair = self.find_max_pair()

            current_vocab_size = len(self.vocab)
            if current_vocab_size % 100 == 0:
                print(f"    - 当前词表大小: {current_vocab_size}/{self.vocab_size}")

            if len(self.vocab) == self.vocab_size or len(max_pair) == 0:
                # self.print_statistics(20)
                # print(self.merges)
                # print(self.vocab)
                print(">>> 达到目标词表大小或无更多可合并项，停止训练。")
                break

            self.still_exist_multi_token = max_pair[0][1] > 1
            self.append_merge(max_pair[0][0])
            new_token = b''.join(max_pair[0][0])
            self.append_vocab(new_token)
            self.update_pairs_in_words(max_pair[0][0])
            # self.print_statistics()

            if self.still_exist_multi_token == False:
                break

        if save:
            self.save_trained_bpe()

    def print_statistics(self, index=3):
        print("====INFO===== Displaying first 5 statistics items")
        for i, (k, v) in enumerate(self.statistics.items()):
            if i >= index:
                break
            print(f"Token: {repr(k)}")
            pprint.pprint(v)
            print("-" * 20)
        print("\n\n")

    def save_trained_bpe(self, path="tokenizer.pkl"):
        if len(self.merges) < 1:
            print("====ERROR===== bpe Never Trained")
        data = {
            "merges": self.merges,
            "vocab": self.vocab
        }
        with open(path, "wb") as f:
            print("save merges:", len(self.merges))
            print("save vocab:", len(self.vocab))
            pickle.dump(data, f)

    def load_trained_bpe(self, path="tokenizer.pkl"):
        with open(path, "rb") as f:
            data = pickle.load(f)
            print("load merges:", len(data["merges"]))
            print("load vocab:", len(data["vocab"]))
            self.vocab = data["vocab"]
            self.merges = data["merges"]

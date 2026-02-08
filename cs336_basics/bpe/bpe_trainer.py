import pprint
from collections import Counter
from collections import defaultdict

times = 'times'
pairs = 'pairs'
tokens = 'tokens'


class bpe_trainer:
    def __init__(self,
                 vocab_size: int,
                 special_tokens: list[str],
                 pat_string: dict):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.special_tokens_len = len(special_tokens)
        self.pat_string = pat_string
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges: list[tuple[bytes, bytes]] = []

        self.still_exist_multi_token = True

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

        return self.statistics

    def update_pairs_in_words(self, max_pair: tuple):
        new_token = b''.join(max_pair)
        affected_words = self.pair_to_statistics[max_pair]

        for k in affected_words:
            v = self.statistics[k]
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

                    if i < len(cur_tokens)-2:
                        right_new_pair = (new_token, cur_tokens[i+2])
                        right_old_pair = (cur_tokens[i+1], cur_tokens[i+2])
                        cur_pairs[right_old_pair] -= 1
                        cur_pairs[right_new_pair] += 1
                        self.pair_to_statistics[right_new_pair].add(k)

                    result_tokens.append(new_token)
                    i += 2
                else:
                    result_tokens.append(cur_tokens[i])
                    i += 1

            v[tokens] = result_tokens
            del cur_pairs[max_pair]
            v[pairs] = +cur_pairs

        del self.pair_to_statistics[max_pair]

    def find_max_pair(self):
        tmp = Counter()
        for _, (_, v) in enumerate(self.statistics.items()):
            for pair in v[pairs]:
                tmp[pair] += v[pairs][pair] * v[times]
        max_freq = max(tmp.values())

        best_pairs = [p for p, f in tmp.items() if f == max_freq]
        best_pairs.sort(reverse=True)

        return [(best_pairs[0], max_freq)]

    def run(self):
        self.still_exist_multi_token = False
        self.global_initialization()
        for token in self.special_tokens:
            self.append_vocab(token.encode())

        while True:
            # i -= 1
            max_pair = self.find_max_pair()
            # print("max_pair:", max_pair)
            # self.print_statistics()

            if len(self.vocab) == self.vocab_size or len(max_pair) == 0:
                # self.print_statistics(20)
                # print(self.merges)
                # print(self.vocab)
                break

            self.still_exist_multi_token = max_pair[0][1] > 1
            self.append_merge(max_pair[0][0])
            new_token = b''.join(max_pair[0][0])
            self.append_vocab(new_token)
            self.update_pairs_in_words(max_pair[0][0])
            # self.print_statistics()

            if self.still_exist_multi_token == False:
                # self.print_statistics(20)
                # print(self.merges)
                # print(self.vocab)
                break

    def print_statistics(self, index=3):
        print("======Displaying first 5 statistics items======")
        for i, (k, v) in enumerate(self.statistics.items()):
            if i >= index:
                break
            print(f"Token: {repr(k)}")
            pprint.pprint(v)
            print("-" * 20)
        print("\n\n")

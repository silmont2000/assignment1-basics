from collections import Counter
from collections import defaultdict
import pickle

times = 'times'
pairs = 'pairs'
tokens = 'tokens'
STRAT_OFFSET = 256


def statistics_initialization(pat_string: dict):
    statistics = {}
    pair_to_statistics = defaultdict(set)
    for k, v in pat_string.items():
        if len(k) < 2:
            statistics[k] = {
                times: v,
                pairs: Counter(),
                tokens: [i for i in k]
            }
            continue

        statistics[k] = {
            times: v,
            pairs: Counter(),
            tokens: [i for i in k]

        }
        for i in range(len(k)-1):
            cur_pair = (k[i], k[i+1])
            statistics[k][pairs][cur_pair] += 1
            pair_to_statistics[cur_pair].add(k)

    return statistics, pair_to_statistics


def load_trained_bpe(path="tokenizer.pkl"):
    with open(path, "rb") as f:
        data = pickle.load(f)
        print("load merges:", len(data["merges"]))
        print("load vocab:", len(data["vocab"]))
        return data

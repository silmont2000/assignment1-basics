import sys
import os

from cs336_basics.token.executor import tokenizer


input_path = 'assignment1-basics/data/small_story.txt'
f = open(input_path, "r", encoding="utf-8")

a_tokenizer = tokenizer(
    text_string=f.read(),
    vocab_size=500,
    special_tokens=["<|endoftext|>"],)
a_tokenizer.run()

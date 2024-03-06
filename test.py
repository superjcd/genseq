from genseq.tools.setup import setup_all
from genseq.tokenizer import NewTokenizer
from genseq.tokenizer.options import TokenizerOptions
from genseq.vocab import NewVocab

setup_all()
tokenizer = NewTokenizer("en", TokenizerOptions(ngrams=1))

tokens = tokenizer.tokenize("i like , coding")
print(tokens)

# torchtext

vocab = NewVocab()

def get_iter(l):
    yield from l 

vocab.build(get_iter(tokens), min_freq=1)

print("i" in vocab)
print(vocab["<unk>"])
print("you" in vocab)

print(vocab.lookup_tokens(vocab.lookup_indices(tokens)))




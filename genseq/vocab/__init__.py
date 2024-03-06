from .vocab_abc import Vocab
from .torchtext import TorchTextVocab



def NewVocab() -> Vocab:
    vocab = TorchTextVocab()
    return vocab 
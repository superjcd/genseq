from .vocab_abc import Vocab
from .torchtext import TorchTextVocab
from .options import VocabOptions


def NewVocab(options: VocabOptions) -> Vocab:
    vocab = TorchTextVocab(options)
    return vocab 
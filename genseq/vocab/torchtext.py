import torchtext
import logging
from typing import List
from .vocab_abc import Vocab
from .special_tokens import *

class TorchTextVocab(Vocab):
    def __init__(self):
        self._vocab = None

    def build(self, samples, min_freq=1, specials: List[str]=SPECIAL_TOKENS):
        self._vocab =  torchtext.vocab.build_vocab_from_iterator(
            samples,
            min_freq,
            specials=specials,
        )
        self._set_unk_index()
        logging.info("vocab build completed")

    def lookup_tokens(self, tokens:List[str]) -> List[int]:
        return self._vocab.lookup_tokens(tokens)
    
    def lookup_indices(self, indcies:List[int]) -> List[str]:
        return self._vocab.lookup_indices(indcies)

    def __contains__(self, key):
        if self._vocab  == None:
            raise Exception("vocab not ready, pleaase run build first")
        return key in self._vocab
    
    def __getitem__(self, key):
        return self._vocab[key]

    def _set_unk_index(self,unk_token=unk_token):
        if self._vocab  == None:
            raise Exception("vocab not ready, pleaase run build first")       
        unk_index = self._vocab[unk_token]
        self._vocab.set_default_index(unk_index)
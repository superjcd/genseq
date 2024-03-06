import torchtext
import logging
from typing import List
from .vocab_abc import Vocab
from .options import VocabOptions

class TorchTextVocab(Vocab):
    def __init__(self, options:VocabOptions=None):
        self._vocab = None
        self._options = options

    def build(self, samples):
        # breakpoint()
        self._vocab =  torchtext.vocab.build_vocab_from_iterator(
            samples,
            min_freq=self._options.min_freq,
            specials=self._options.special_tokens, 
        )
        self._set_unk_index()
        logging.info(f"vocab: {self._options.name}  build completed")

    def lookup_tokens(self, tokens:List[str]) -> List[int]:
        return self._vocab.lookup_tokens(tokens)
    
    def lookup_indices(self, indcies:List[int]) -> List[str]:
        return self._vocab.lookup_indices(indcies)

    def __contains__(self, key):
        if self._vocab  == None:
            raise Exception("vocab not ready, please run build first")
        return key in self._vocab
    
    def __getitem__(self, key):
        return self._vocab[key]

    def _set_unk_index(self):
        if self._vocab  == None:
            raise Exception("vocab not ready, please run build first")       
        unk_index = self._vocab[self._options.unk_token]
        self._vocab.set_default_index(unk_index)
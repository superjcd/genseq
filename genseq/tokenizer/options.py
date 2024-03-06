from dataclasses import dataclass

@dataclass
class TokenizerOptions:
    ngrams:int = 1
    keep_stop_words: bool = True
    keep_punct: bool = True
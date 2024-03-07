from dataclasses import dataclass, field
from typing import List 
from .special_tokens import *

@dataclass
class VocabOptions:
    name: str = "vocab"
    min_freq:int =1
    unk_token :str = unk_token
    pad_token :str = pad_token
    sos_token :str = sos_token
    eos_token :str = eos_token
    special_tokens: List[str]=field(default_factory=list)

    def __post_init__(self):
        assert len(self.special_tokens) == 4
        assert self.unk_token in self.special_tokens
        assert self.pad_token in self.special_tokens
        assert self.sos_token in self.special_tokens
        assert self.eos_token in self.special_tokens
from dataclasses import dataclass, field
from typing import List 
from .special_tokens import SPECIAL_TOKENS, unk_token

@dataclass
class VocabOptions:
    name: str = "vocab"
    min_freq:int =1
    special_tokens: List[str]=field(default_factory=list)
    unk_token :str = unk_token
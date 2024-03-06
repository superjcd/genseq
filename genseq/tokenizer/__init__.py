from .tokenizer_abc import Tokenizer 
from .spacy import SpacyTokenizer
from .options import TokenizerOptions


def NewTokenizer(language:str, options:TokenizerOptions) -> Tokenizer:
    return SpacyTokenizer(language, options)
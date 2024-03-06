import datasets
from dataclasses import dataclass 
from typing import Dict
from genseq.tokenizer import Tokenizer

dataset = datasets.load_datasets()


@dataclass 
class Options:
    tokenize: bool = False 
    feature_tokenizer_mapping: Dict[str, Tokenizer] = {}
    add_vocab_index: bool = False
    


def NewDataSet(name, split, tokenizer, vocab):
    ...


    
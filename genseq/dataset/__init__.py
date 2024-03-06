import datasets
from dataclasses import dataclass, field
from typing import Dict
from genseq.tokenizer import Tokenizer
from genseq.vocab import Vocab
from genseq.vocab.special_tokens import *

@dataclass 
class DatasetOptions:
    tokenize: bool = False 
    feature_tokenizer_mapping: Dict[str, Tokenizer] = field(default_factory=dict)
    max_length: int = 100
    lower: bool = True
    sos_token: str = sos_token
    eos_token: str = eos_token
    index: bool = False
    feature_vocab_mapping:Dict[str, Vocab] = field(default_factory=dict)

def tokenize_example(example, feature:str,tokenizer:Tokenizer, options:DatasetOptions):
    tokens = tokenizer.tokenize(example[feature])[:options.max_length]
    if options.lower:
        tokens = [token.lower() for token in tokens]
    tokens = [sos_token] + tokens + [eos_token]
    feature_key = feature + "_" + "tokens" 
    return {feature_key: tokens}

def index_example(example, feature:str, vocab:Vocab):
    tokens_key = feature + "_" + "tokens" 
    ids = vocab.lookup_indices(example[tokens_key])
    feature_key = feature + "_" + "ids" 
    return {feature_key: ids}

def NewDataSet(path, split, options:DatasetOptions=None):
    dataset = datasets.load_dataset(path, split=split)
    if options.tokenize:
        for feature, tokenizer in options.feature_tokenizer_mapping.items():
            if feature not in dataset.features:
                raise IndexError(f"{feature} not in dataset")
            dataset = dataset.map(tokenize_example, fn_kwargs={"feature":feature, "tokenizer":tokenizer, "options": options})
    if options.index:
        for feature, vocab in options.feature_vocab_mapping.items():
            if feature not in dataset.features:
                raise IndexError(f"{feature} not in dataset")
            vocab.build(dataset[f"{feature}_tokens"])
            dataset = dataset.map(index_example, fn_kwargs={"feature":feature, "vocab": vocab})
    return dataset


    
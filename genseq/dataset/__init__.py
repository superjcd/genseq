import datasets
import torch
from torch import nn
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
    to_torch:bool = False

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

        if options.to_torch:
            cols = []
            for feature in options.feature_vocab_mapping.keys():
                feature_key = feature + "_" + "ids" 
                cols.append(feature_key)
            dataset = dataset.with_format(
                        type="torch",
                        columns=cols,
                        output_all_columns=True,
                    )

    return dataset


def get_collate_fn(features, pad_index: int):
    def collate_fn(batch):
        new_batch = {}
        for feature in features:
            key = feature + "_" + "ids" 
            data = [example[key] for example in batch]
            data = nn.utils.rnn.pad_sequence(data, padding_value=pad_index)
            new_batch[key] = data
        return new_batch
    return collate_fn


def NewDataLoader(dataset, batch_size, features, pad_index, shuffle=False):
    collate_fn = get_collate_fn(features, pad_index)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle,
    )
    return data_loader
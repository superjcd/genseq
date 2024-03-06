from genseq.tools.setup import setup_all
from genseq.tokenizer import NewTokenizer
from genseq.tokenizer.options import TokenizerOptions
from genseq.vocab import NewVocab
from genseq.vocab.options import VocabOptions
from genseq.vocab.special_tokens import SPECIAL_TOKENS
from genseq.dataset import NewDataSet, DatasetOptions

setup_all()
tokenizer_en = NewTokenizer("en", TokenizerOptions(ngrams=1))
tokenizer_de = NewTokenizer("de", TokenizerOptions(ngrams=1))


vocab_en_options = VocabOptions(name="vocab_en", min_freq=2, special_tokens=SPECIAL_TOKENS)
vocab_en = NewVocab(vocab_en_options)
vocab_de_options = VocabOptions(name="vocab_de", min_freq=2, special_tokens=SPECIAL_TOKENS)
vocab_de = NewVocab(vocab_de_options)


options = DatasetOptions(tokenize=True, 
                         feature_tokenizer_mapping={"en": tokenizer_en, "de": tokenizer_de}, 
                         max_length=1000,
                         lower=True,
                         index=True,
                         feature_vocab_mapping={"en":vocab_en, "de":vocab_de})
train_dataset = NewDataSet("bentrevett/multi30k", "validation", options=options)

print(train_dataset)
print(train_dataset.features)

print(train_dataset[0]["en_tokens"])
print(train_dataset[0]["en_ids"])


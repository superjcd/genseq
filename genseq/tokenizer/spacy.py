import sys
import spacy
from typing import List
from typing import Callable
from spacy.tokens import Token
from termcolor import colored
from .tokenizer_abc import Tokenizer
from .options import TokenizerOptions

class SpacyTokenizer(Tokenizer):
    def __init__(
        self,
        language: str,
        options:TokenizerOptions
    ):
        """
        Parameters
        ----
        language:language  abbreviation, 'en', 'de' etc. use `supported_languages` to get all supported languages
        ngrams: ngarm, 1 means tokenize sequence to one word, 2 to two words
        keep_stop_words:  keep stop words ot not 
        keep_punct: keep punctuation or not 
        """
        if language not in self.supported_languages:
            raise ValueError(
                f"Language `{language}` is not supported, supported languages are:{','.join(self.supported_languages)}"
            )
        self.language = language
        self.nlp = self.prepare_model_by_language(language)
        self.tokenizer = self.nlp.tokenizer
        self.ngrams = options.ngrams
        self.keep_stop_words = options.keep_stop_words
        self.keep_punct = options.keep_stop_words
        self.custom_extensions = set()

    def tokenize(self, string) -> List[str]:
        tokens = self.tokenizer(string)  
        tokens = [t for t in tokens if self._satisfy(t)]
        raw_tokens = [t.text for t in tokens]
        if self.ngrams == 1:
            return raw_tokens
        
        total_tokens = len(raw_tokens)
        result = []
        for i, _ in enumerate(raw_tokens):
            if i <= total_tokens - self.ngrams:
                result.append(" ".join((raw_tokens[i : (i + self.ngrams)])))
        return result

    def prepare_model_by_language(self, language):
        if language in ["en", "zh"]:
            model = f"{language}_core_web_sm"
        else:
            model = f"{language}_core_news_sm"
        try:
            loaded = spacy.load(model)
        except OSError:
            print(colored(f"language model for `{language}` not found, use `python -m spacy download {model}`to download", "red"))
            sys.exit()
        else:
            return loaded
    
    def register_extension(self, name, extension: Callable):
        if callable(extension):
            Token.set_extension(name, getter=extension)
            self.custom_extensions.add(name)
            print(f"custome token extension function  `{name}` registered successfully")

        else:
            raise TypeError(f"`extension` shoud be a callable")

    def remove_extension(self, name):
        Token.remove_extension(name)
        self.custom_extensions.remove(name)

    def _satisfy(self, token: Token) -> bool:
        condition = True
        
        if not self.keep_stop_words:
            condition = condition & (not token.is_stop)

        if not self.keep_punct:
            condition = condition & (not token.is_punct)

        for extention in self.custom_extensions:
            condition = condition & getattr(token._, extention)

        return condition

    @property
    def supported_languages(self):
        return ["en", "es", "fr", "de", "sv", "nl", "pl", "ja", "zh"]

    def __call__(self, text):
        return self.tokenize(text)
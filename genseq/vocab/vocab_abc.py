from abc import ABC, abstractmethod, abstractproperty

class Vocab(ABC):
    @abstractmethod
    def build():
        ...

    @abstractmethod
    def lookup_tokens():
        ...

    @abstractmethod
    def lookup_indices():
        ...
    
    @abstractproperty
    def pad_token_index():
        ...

    @abstractproperty
    def unk_token_index():
        ...


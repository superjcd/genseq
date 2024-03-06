from abc import ABC, abstractmethod

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




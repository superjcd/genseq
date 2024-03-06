from abc import ABC 
from abc import abstractmethod
from typing import List



class Tokenizer(ABC):
    @abstractmethod
    def tokenize(self, string:str) -> List[str] :
        ...

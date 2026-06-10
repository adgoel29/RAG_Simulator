from abc import ABC, abstractmethod

class VectorStoreBase(ABC):

    @abstractmethod
    def build_index(self, chunks: list[str]):
        ...

    @abstractmethod
    def search(self, query: str, method: str, topk: int) -> list:
        ...

    @abstractmethod
    def supported_methods(self) -> list[str]:
        ...
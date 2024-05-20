from abc import ABC, abstractmethod

from llama_index.core import StorageContext
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import Document
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from omegaconf import DictConfig


class BaseQueryEngineFactory(ABC):
    @classmethod
    @abstractmethod
    def create_engine(
        cls,
        docs: list[Document],
        storage_context: StorageContext,
        cfg: DictConfig,
        postprocessors: list[BaseNodePostprocessor] = [],
    ) -> RetrieverQueryEngine:
        pass

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import Document
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from omegaconf import DictConfig
from .base_engine import BaseQueryEngineFactory


class BasicQueryEngineFactory(BaseQueryEngineFactory):
    @classmethod
    def create_engine(
        cls,
        docs: list[Document],
        storage_context: StorageContext,
        cfg: DictConfig,
        postprocessors: list[BaseNodePostprocessor] = [],
    ) -> RetrieverQueryEngine:
        node_parser = SimpleNodeParser.from_defaults()
        nodes = node_parser.get_nodes_from_documents(docs)
        index = VectorStoreIndex(nodes, storage_context=storage_context, show_progress=cfg.verbose)

        if cfg.hybrid:
            return index.as_query_engine(
                similarity_top_k=cfg.similarity_top_k,
                sparse_top_k=12,
                vector_store_query_mode="hybrid",
                node_postprocessors=postprocessors,
                alpha=0.5
            )  # type: ignore

        return index.as_query_engine(
            similarity_top_k=cfg.similarity_top_k, node_postprocessors=postprocessors
        )  # type: ignore

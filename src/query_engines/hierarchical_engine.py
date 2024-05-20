from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.schema import Document
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from omegaconf import DictConfig
from .base_engine import BaseQueryEngineFactory


class HierarchicalQueryEngineFactory(BaseQueryEngineFactory):
    @classmethod
    def create_engine(
        cls,
        docs: list[Document],
        storage_context: StorageContext,
        cfg: DictConfig,
        postprocessors: list[BaseNodePostprocessor] = [],
    ) -> RetrieverQueryEngine:
        node_parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=[n * cfg.chunk_size for n in reversed(range(1, 4))], chunk_overlap=cfg.chunk_overlap
        )
        nodes = node_parser.get_nodes_from_documents(docs)
        index = VectorStoreIndex(nodes, storage_context=storage_context, store_nodes_override=True)

        if cfg.hybrid:
            base_retriever = index.as_retriever(
                similarity_top_k=cfg.similarity_top_k, sparse_top_k=12, vector_store_query_mode="hybrid", alpha=0.5
            )
        else:
            base_retriever = index.as_retriever(similarity_top_k=cfg.similarity_top_k)

        retriever = AutoMergingRetriever(
            base_retriever,  # type: ignore
            storage_context=index.storage_context,
            verbose=cfg.verbose,
        )
        return RetrieverQueryEngine.from_args(retriever, node_postprocessors=postprocessors)

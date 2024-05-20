from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter, SentenceWindowNodeParser

from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import Document
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from omegaconf import DictConfig
from .base_engine import BaseQueryEngineFactory


class SentenceWindowQueryEngineFactory(BaseQueryEngineFactory):
    @classmethod
    def create_engine(
        cls,
        docs: list[Document],
        storage_context: StorageContext,
        cfg: DictConfig,
        postprocessors: list[BaseNodePostprocessor] = [],
    ) -> RetrieverQueryEngine:
        node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=1,
            window_metadata_key="window",
            original_text_metadata_key="original_text",
            sentence_splitter=SentenceSplitter(chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap).split_text,
        )
        nodes = node_parser.get_nodes_from_documents(docs)
        index = VectorStoreIndex(nodes, storage_context=storage_context, show_progress=cfg.verbose)

        postprocessors = [MetadataReplacementPostProcessor(target_metadata_key="window")] + postprocessors
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

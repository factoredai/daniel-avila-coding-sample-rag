from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.postprocessor.colbert_rerank import ColbertRerank
from llama_index.core.postprocessor import LongContextReorder
from omegaconf import DictConfig
from .base_engine import BaseQueryEngineFactory
from .basic_engine import BasicQueryEngineFactory
from .hierarchical_engine import HierarchicalQueryEngineFactory
from .sentence_window_engine import SentenceWindowQueryEngineFactory


def create_query_engine(docs, storage_context, cfg: DictConfig) -> RetrieverQueryEngine:
    engines_dict = {
        "basic": BasicQueryEngineFactory,
        "hierarchical": HierarchicalQueryEngineFactory,
        "sentence_window": SentenceWindowQueryEngineFactory,
    }

    postprocessors = []
    if cfg.reranker:
        reranker = ColbertRerank(
            top_n=3,
            model="colbert-ir/colbertv2.0",
            tokenizer="colbert-ir/colbertv2.0",
            device="cpu",
            keep_retrieval_score=True,
        )
        postprocessors.append(reranker)

    if cfg.reorder:
        reorder = LongContextReorder()
        postprocessors.append(reorder)

    engine_factory: BaseQueryEngineFactory = engines_dict.get(cfg.engine_type)  # type: ignore
    if not engine_factory:
        raise ValueError(f"Unknown engine. Available engine types are: {', '.join(list(engines_dict.keys()))}")
    return engine_factory.create_engine(docs, storage_context, cfg, postprocessors)

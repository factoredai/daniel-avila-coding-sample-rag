from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llama_cpp.llama_utils import completion_to_prompt, messages_to_prompt
from omegaconf import DictConfig
from transformers import AutoTokenizer
from llama_index.core import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient


def llama_index_setup(cfg: DictConfig):
    Settings.embed_model = HuggingFaceEmbedding(model_name=cfg.embedding_model, device="cpu")
    Settings.tokenizer = AutoTokenizer.from_pretrained(cfg.embedding_model)  # type: ignore
    Settings.llm = LlamaCPP(
        model_path=cfg.llm_path,
        temperature=0.0,
        max_new_tokens=cfg.max_tokens,
        context_window=cfg.n_batch,
        generate_kwargs={},
        model_kwargs={"n_gpu_layers": cfg.n_gpu_layers},
        # transform inputs into Llama2 format
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        verbose=False,
    )

    if cfg.debug_mode:
        llama_debug = LlamaDebugHandler(print_trace_on_end=True)
        callback_manager = CallbackManager([llama_debug])
        Settings.callback_manager = callback_manager


def qdrant_setup(cfg) -> StorageContext:
    client = QdrantClient(url=cfg.qdrant_url)
    client.delete_collection(cfg.collection_name)

    vector_store = QdrantVectorStore(client=client, collection_name=cfg.collection_name, enable_hybrid=cfg.hybrid)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return storage_context

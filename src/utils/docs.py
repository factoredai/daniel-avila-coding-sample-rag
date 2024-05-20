import os
from datasets import load_dataset
from llama_index.core import Document


def get_pdf_docs(data_dir, reader):
    files_paths = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    docs = []
    for f in files_paths:
        _docs = reader.load(os.path.join(data_dir, f), metadata=True)
        docs.extend(_docs)
    return docs


def get_hf_docs(cfg):
    dataset = load_dataset(cfg.dataset_name, name=cfg.file, cache_dir=cfg.data_cache_dir, trust_remote_code=True)
    docs = [Document(text=text) for text in dataset["train"]["text"] + dataset["validation"]["text"]]  # type: ignore
    return docs

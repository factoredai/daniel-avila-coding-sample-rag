from typing import Union
from dotenv import load_dotenv
import tiktoken
from tiktoken.core import Encoding
assert load_dotenv("src/envs/.env"), "The env was  not found."
from llama_index.core import Settings
from datetime import datetime
import mlflow
import hydra
from llama_index.llms.openai import OpenAI
from omegaconf import DictConfig
from transformers import AutoTokenizer, PreTrainedTokenizer
from src.setup import llama_index_setup, qdrant_setup
from src.utils.docs import get_hf_docs
from src.testers.needle_tester import LLMNeedleHaystackTester, Needle
from llama_index.core.llms.llm import LLM


def create_tested_llm(cfg: DictConfig) -> LLM:
    llm_dict = {
        "default": Settings.llm,
        "gpt-3.5-turbo": OpenAI("gpt-3.5-turbo"),
        "gpt-4": OpenAI("gpt-4"),
        "gpt-4-turbo": OpenAI("gpt-4-turbo"),
    }
    try:
        llm = llm_dict[cfg.tested_llm_name]
        return llm
    except KeyError:
        raise KeyError(f"Unknown LLM. Available engine types are: {', '.join(list(llm_dict.keys()))}")


def create_tokenizer(cfg: DictConfig) -> Union[PreTrainedTokenizer, Encoding]:
    toeknizer_dict = {
        "default": AutoTokenizer.from_pretrained("meta-llama/Llama-3-8b-hf"),
        "gpt-3.5-turbo": tiktoken.encoding_for_model("gpt-3.5-turbo"),
        "gpt-4": tiktoken.encoding_for_model("gpt-4"),
        "gpt-4-turbo": tiktoken.encoding_for_model("gpt-4"),
    }
    try:
        tokenizer = toeknizer_dict[cfg.tested_llm_name]
        return tokenizer
    except KeyError:
        raise KeyError(f"Unknown LLM. Available engine types are: {', '.join(list(toeknizer_dict.keys()))}")


@hydra.main(version_base=None, config_path="src/config", config_name="config")
def main(cfg: DictConfig):
    # LLama-index Setup
    llama_index_setup(cfg.llm)

    # Qdrant Client Setup}
    storage_context = None
    if cfg.haystack.rag_haystack:
        storage_context = qdrant_setup(cfg.storage_context)

    # Load Documents
    docs = get_hf_docs(cfg.dataset)

    mlflow.set_experiment(cfg.haystack.experiment_name)
    with mlflow.start_run(run_name=cfg.haystack.run_name + f"-ts:{datetime.today().timestamp()}"):
        evaluation_llm = OpenAI("gpt-4")
        tested_llm = create_tested_llm(cfg.haystack)
        tokenizer = create_tokenizer(cfg.haystack)
        context_lengths = [1000, 10000, 100000]
        depth_percents = [0, 50, 100]
        needle = Needle("What is the secret ingredient?", "The secret ingredient is Tomato.")

        needle_tester = LLMNeedleHaystackTester(
            tested_llm,
            evaluation_llm,
            tokenizer,  # type: ignore
            docs,
            needle,
            context_lengths,
            depth_percents,
            cfg.haystack,
            cfg.query_engine,
            storage_context
        )

        avg_result = needle_tester.run_test()

        # Log into Mlflow
        for key, value in cfg.haystack.items():
            mlflow.log_param(str(key), value)

        for result in avg_result:
            mlflow.log_metric(key=result["context_lengths"], value=result["avg_score"])


if __name__ == "__main__":
    main()

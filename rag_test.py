import asyncio
import os
import pickle

from dotenv import load_dotenv

assert load_dotenv("src/envs/.env"), "The env was  not found."

from datetime import datetime
import mlflow
import hydra
from llama_index.llms.openai import OpenAI
from omegaconf import DictConfig

from src.query_engines.engine_factory import create_query_engine
from src.setup import llama_index_setup, qdrant_setup
from src.utils.docs import get_hf_docs
from testers.rag_tester import create_testset, RAGTester


@hydra.main(version_base=None, config_path="src/config", config_name="config")
def main(cfg: DictConfig):
    # LLama-index Setup
    llama_index_setup(cfg.llm)

    # Qdrant Client Setup
    storage_context = qdrant_setup(cfg.storage_context)

    # Load Documents
    docs = get_hf_docs(cfg.dataset)

    mlflow.set_experiment(cfg.general.experiment_name)
    with mlflow.start_run(run_name=cfg.general.run_name + f"-ts:{datetime.today().timestamp()}"):
        # Create Query Engine
        query_engine = create_query_engine(docs, storage_context, cfg.query_engine)

        # Get or Create TestSet
        if not os.path.isfile(cfg.general.testset_path) or cfg.general.regenerate_testset:
            generator = OpenAI(model="gpt-3.5-turbo", temperature=0)
            testset = asyncio.run(create_testset(generator, docs, cfg.general.test_size_per_doc))
            with open(cfg.general.testset_path, "wb") as f:
                pickle.dump(testset, f)
        else:
            with open(cfg.general.testset_path, "rb") as f:
                testset = pickle.load(f)

        # Evaluate RAG System
        tester = RAGTester(testset, query_engine)
        result = tester.run_test()

        # Log into Mlflow
        for key, value in cfg.query_engine.items():
            mlflow.log_param(str(key), value)
        mlflow.log_metrics(result.copy())


if __name__ == "__main__":
    main()

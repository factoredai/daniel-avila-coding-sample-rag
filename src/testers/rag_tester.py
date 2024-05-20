from typing import Optional
from datasets import Dataset
from llama_index.core import Document
from llama_index.core.evaluation import DatasetGenerator, QueryResponseDataset
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.openai import OpenAI
from ragas.evaluation import Result, evaluate
from ragas.metrics.base import Metric
from ragas.metrics import (
    answer_correctness,
    answer_relevancy,
    answer_similarity,
    context_precision,
    context_recall,
    faithfulness,
)
from tqdm import tqdm
from base_tester import BaseTester


async def create_testset(generator: OpenAI, docs: list[Document], test_size: int) -> QueryResponseDataset:
    test_generator = DatasetGenerator.from_documents(documents=docs, llm=generator)
    testset = test_generator.agenerate_dataset_from_nodes(num=test_size)
    return await testset


class RAGTester(BaseTester):
    def __init__(
        self, testset: QueryResponseDataset, query_engine: RetrieverQueryEngine, metrics: Optional[list[Metric]] = None
    ):
        if not metrics:
            self.metrics = [
                faithfulness,  # Evaluates faithfulness of the response to the source material.
                answer_relevancy,  # Assesses relevance of the response to the query.
                context_precision,  # Measures precision of the context in the response.
                context_recall,  # Measures recall of the context in the response.
                answer_correctness,  # Checks correctness of the answer.
                answer_similarity,  # Evaluates similarity of the answer to a reference answer.
            ]
        else:
            self.metrics = metrics

        self.testset = testset
        self.query_engine = query_engine

    def run_test(self) -> Result:
        test_questions, test_answers = list(zip(*self.testset.qr_pairs))
        responses = [self.query_engine.query(q) for q in tqdm(test_questions, desc="Generating responses...")]
        answers = []
        contexts = []
        for r in responses:
            answers.append(r.response)  # type: ignore
            contexts.append([c.node.get_content() for c in r.source_nodes])

        dataset_dict = {
            "question": list(test_questions),
            "answer": answers,
            "contexts": contexts,
            "ground_truth": list(test_answers),
        }

        ds = Dataset.from_dict(dataset_dict)
        result = evaluate(ds, self.metrics, is_async=True)
        return result

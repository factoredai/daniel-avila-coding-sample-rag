# This code was adapted from gkamradt LLMTest_NeedleInAHaystack repository, avilable on
# https://github.com/gkamradt/LLMTest_NeedleInAHaystack under the MIT License.
# MIT License
# Copyright (c) 2023 Greg Kamradt

from typing import Optional

import os
import asyncio
from tqdm import tqdm
from itertools import groupby
from llama_index.core import Document
from dataclasses import dataclass
from transformers import PreTrainedTokenizer
from llama_index.core.evaluation import CorrectnessEvaluator
from llama_index.core.llms.llm import LLM
from llama_index.core import StorageContext

from ..query_engines.engine_factory import create_query_engine
from omegaconf import DictConfig
import json
from llama_index.core.llms import ChatMessage, MessageRole
from .base_tester import BaseTester


def get_llm_answer(context: str, question: str, llm: LLM) -> str:
    system_prompt = """\
    You are a helpful AI bot that answers questions for a user. Keep your response short and direct.
    """

    context_prompt = f"""\
    CONTEXT:
    {context}

    -------------------------

    {question} - Don't give information outside the context or repeat the instruction.
    """

    message_templates = [
        ChatMessage(content=system_prompt, role=MessageRole.SYSTEM),
        ChatMessage(content=context_prompt, role=MessageRole.USER,),
    ]
    response = asyncio.run(llm.achat(message_templates))
    return response.message.content.strip()  # type: ignore


@dataclass
class Needle:
    query: str
    answer: str

    def __str__(self):
        return self.query + " " + self.answer


class LLMNeedleHaystackTester(BaseTester):
    MAX_SCORE = 5

    def __init__(
        self,
        tested_llm: LLM,
        evaluation_llm: LLM,
        tokenizer: PreTrainedTokenizer,
        docs: list[Document],
        needle: Needle,
        context_lengths: list[int],
        depth_percents: list[int],
        cfg: DictConfig,
        query_engine_cfg: Optional[DictConfig] = None,
        storage_context: Optional[StorageContext] = None,
        final_context_length_buffer: int = 30,
        save_results: bool = True,
        seconds_to_sleep_between_completions: int = 5,
    ):
        assert not cfg.rag_haystack or storage_context, "Must specify RAGs' StorageContext"
        assert not cfg.rag_haystack or query_engine_cfg, "Must specify RAGs' Configuration"

        self.test_description = cfg.test_description
        self.tested_llm = tested_llm
        self.tokenizer = tokenizer
        self.docs = docs
        self.context_lengths = context_lengths
        self.depth_percents = depth_percents
        self.needle = needle
        self.evaluator = CorrectnessEvaluator(llm=evaluation_llm)
        self.final_context_length_buffer = final_context_length_buffer
        self.version = cfg.version
        self.cfg = cfg
        self.save_results = save_results
        self.seconds_to_sleep_between_completions = seconds_to_sleep_between_completions

        self.query_engine_cfg = query_engine_cfg
        self.storage_context = storage_context

        self.full_context = self.read_docs()

        self.results_dir = cfg.haystack_results_dir
        self.results_path = os.path.join(self.results_dir, f"{self.test_description}_results.json")

    def run_test(self) -> list[dict]:
        for context_length in tqdm(self.context_lengths, position=0, disable=not self.cfg.verbose):
            for depth_percent in tqdm(self.depth_percents, position=1, disable=not self.cfg.verbose):
                self.evaluate_and_log(context_length, depth_percent)

        results = self._get_results()
        return self._get_avg_results(results)

    def evaluate_and_log(self, context_length: int, depth_percent: int):
        results = self._get_results()

        if self._result_exists(results, context_length, depth_percent):
            return None

        context = self.generate_context(context_length, depth_percent)

        if self.cfg.rag_haystack:
            docs = [Document(text=context)]
            retriever = create_query_engine(docs, self.storage_context, self.query_engine_cfg).retriever  # type: ignore
            nodes = retriever.retrieve(self.needle.query)
            context = "\n\n\n".join([n.node.get_content() for n in nodes])

        response = get_llm_answer(context, self.needle.query, self.tested_llm)

        score = self.evaluate_response(str(response), self.needle.answer, self.needle.query) / self.MAX_SCORE
        result = {
            "context_length": context_length,
            "depth_percent": depth_percent,
            "needle": str(self.needle),
            "response": str(response),
            "score": score,
            "version": self.version,
        }
        results.append(result)
        self._log_results(results)

    def evaluate_response(self, response: str, answer: str, query: str) -> float:
        result = self.evaluator.evaluate(
            query=query,
            response=response,
            reference=answer,
        )
        return result.score  # type: ignore

    def generate_context(self, context_length: int, depth_percent: int) -> str:
        context = self.encode_and_trim(self.full_context, context_length)
        context = self.insert_needle(context, context_length, depth_percent)
        return context

    def read_docs(self):
        context = ""
        n_tokens = 0
        max_context_length = max(self.context_lengths)
        for doc in self.docs:
            context += doc.get_content()
            n_tokens = len(self.tokenizer.encode(context))
            if n_tokens >= max_context_length:
                break
        return context

    def encode_and_trim(self, context: str, context_length: int) -> str:
        trimed_context = context
        tokens = self.tokenizer.encode(context)
        if len(tokens) > context_length:
            trimed_context = self.tokenizer.decode(tokens[:context_length])
        return trimed_context

    def insert_needle(self, context: str, context_length: int, depth_percent: int, ) -> str:
        tokens_needle = self.tokenizer.encode(self.needle.answer)
        tokens_context = self.tokenizer.encode(context)
        context_length -= self.final_context_length_buffer

        if len(tokens_context) + len(tokens_needle) > context_length:
            tokens_context = tokens_context[:context_length - len(tokens_needle)]

        if depth_percent == 100:
            tokens_new_context = tokens_context + tokens_needle
        else:
            insertion_point = int(len(tokens_context) * (depth_percent / 100))
            tokens_new_context = tokens_context[:insertion_point]
            period_tokens = self.tokenizer.encode('.')

            while tokens_new_context and tokens_new_context[-1] not in period_tokens:
                insertion_point -= 1
                tokens_new_context = tokens_context[:insertion_point]

            tokens_new_context += tokens_needle + tokens_context[insertion_point:]

        new_context = self.tokenizer.decode(tokens_new_context)
        return new_context

    def _get_results(self) -> list:
        try:
            with open(self.results_path, "r") as f:
                results = json.load(f)
        except FileNotFoundError:
            results = []
            pass
        return results

    def _log_results(self, results: list[dict]):
        with open(self.results_path, "w") as f:
            json.dump(results, f)

    def _result_exists(self, results: list[dict], context_length: int, depth_percent: int) -> bool:
        conditions_met = []
        for result in results:
            context_length_met = result['context_length'] == context_length
            depth_percent_met = result['depth_percent'] == depth_percent
            version_met = result.get('version', 1) == self.version
            conditions_met.append(context_length_met and depth_percent_met and version_met)
        return any(conditions_met)

    def _get_avg_results(self, results: list[dict]):
        avg_results = []
        for k, v in groupby(results, key=lambda x: x["context_length"]):
            v_list = list(v)
            avg_results.append(
                {"context_length": k, "avg_score": round(sum(int(d["score"]) / len(v_list) for d in v_list), 3)}
            )
        return avg_results

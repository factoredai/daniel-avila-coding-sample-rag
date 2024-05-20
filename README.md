# **RAG Systems Tester**

This repository contains the code to evaluate and experiment with a variety of RAG systems on either a HuggingFace-compatible dataset or a PDF group. RAG systems are implemented using LLamaIndex as an indexing framework and Qdrant as a VectorDB. The latter is configured using a docker container that can be easily set by using the command `docker compose up`. The directory structure is slightly different from the original repository to disclose some additional confidential features.

## RAGAS and Needle-in-a-Haystack Tests

There are two major groups of tests, each implemented in a different script:

1. **General RAG Tests** (see *rag_test.py*): This script performs the standard test on the RAG system, it evaluates both **retrieval** (context precision and recall) and **generation** (faithfulness, answer relevancy), but it also includes comprehensive tests of the system performance (answer semantic similarity and correctness). The evaluation can be performed on a user-provided dataset, but it generates a synthetic dataset by default. All the tests are implemented using RAGAS.
2. **Needle-in-a-Haystack** (see *haystack_test.py*) This test is based on Greg Kamradt's implementation (check his [repo](https://github.com/gkamradt/LLMTest_NeedleInAHaystack)). It introduces little and specific pieces of information (needles) in the documents that are used to build the RAG system and then tests the system's ability to efficiently retrieve those pieces.

## Query Engines

The query engine is the main element of the RAG system, determining not only how the nodes are parsed but also the retrieval logic which ultimately will provide the LLM's context. Although the flexibility of the implementation allows for several different configurations, three main types of engines are used:

1. **Basic Query Engine:** This engine divides the documents by a specified number of tokens without taking into account any additional semantic information. Afterward, it retrieves the top-k most similar fragments and adds its content directly into the LLM context.
2. **Sentence-Window Query Engine:** This engine uses a sentence parser to divide the documents into sentences which are then embedded and included in the VectorDB. Additionally, the parser also adds a window context (i.e. n contiguous sentences) in the metadata of each embedding. Thus, although the similarity measure is done at the sentence level, the LLM context is constructed using the entire window.
3. **Hierarchical Query Engine:** This type of engine dynamically builds the embeddings by splitting a document into a recursive hierarchy. In other words, it creates parent nodes (e.g. with a bigger chunk size), and child nodes per parent (e.g. with a smaller chunk size), and adds the relationship between them as metadata. When retrieving, the engine will try to merge leaf chunks if possible, and return the parent chunk.

## Setup and Experimentation

Most of the configuration is contained in a `.yaml` file which follows Hydra format. Hydra acts as an enhanced command-line parser that facilitates conducting a variety of experiments with different configurations. Additionally, all the experiments are logged via MLflow to keep track of and compare the results.

For confidentiality, some fields of the configuration file have been erased and replaced with "•••".

## Environment

Poetry is used to manage the dependencies in this project.
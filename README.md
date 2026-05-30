# DocIntel – AI Document Assistant

DocIntel is a production-style Retrieval-Augmented Generation (RAG) system that enables users to upload documents, ask natural language questions, and receive grounded answers backed by retrieved evidence.

Unlike basic RAG implementations, DocIntel combines Hybrid Retrieval (BM25 + Vector Search), Cross-Encoder Reranking, Guardrails, Conversation Memory, and a Custom Evaluation Framework to improve retrieval quality, answer relevance, and explainability.

## Key Results

* Achieved **93.3% Retrieval Quality Score** on benchmark evaluation queries.
* Achieved **86.7% Answer Quality Score** on benchmark evaluation queries.
* Average end-to-end response latency of **~2.8 seconds**.
* Integrated **Hybrid Retrieval** using BM25 and semantic vector search for improved recall.
* Implemented **Cross-Encoder Reranking** to improve context relevance before generation.
* Added **Guardrails** to filter unsafe or irrelevant queries.
* Built a custom evaluation framework to measure retrieval quality, answer quality, and latency.
* Provides retrieval diagnostics including confidence score, reranking score, BM25 score, and retrieved evidence visibility.

## Tech Stack

* FastAPI
* Streamlit
* ChromaDB
* Sentence Transformers
* BM25 Retrieval
* Cross-Encoder Reranking
* Groq LLMs
* Python

## Architecture

PDF Upload → Text Extraction → Chunking → Embedding Generation → ChromaDB Storage → Hybrid Retrieval (BM25 + Vector Search) → Cross-Encoder Reranking → Guardrails → LLM Generation → Evaluation & Diagnostics

## Features

* Document Upload & Indexing
* Hybrid Retrieval
* Semantic Search
* Cross-Encoder Reranking
* Conversation Memory
* Guardrails
* Retrieval Diagnostics Dashboard
* Custom Evaluation Framework
* Source Evidence Display
* Configurable LLM Parameters

This project demonstrates practical GenAI engineering concepts commonly used in modern AI-powered search and knowledge assistant systems.

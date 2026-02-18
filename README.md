# GraphRAG-Powered Airline Intelligence Assistant

This project presents an end-to-end Graph Retrieval-Augmented Generation (Graph-RAG) system that combines machine learning, knowledge graphs, and large language models (LLMs) to generate grounded, explainable airline insights. The system integrates structured airline operational data, passenger feedback, and booking information into a unified intelligence pipeline designed to reduce hallucination, improve factual accuracy, and enhance interpretability of AI-generated responses.

The core objective of this project is to demonstrate how symbolic reasoning through a Neo4j Knowledge Graph can collaborate with statistical reasoning from LLMs to produce reliable, transparent, and data-driven answers.

# Porject Overview 

The system begins with large-scale airline datasets containing booking behavior, operational flight details, survey responses, and passenger reviews. The data undergoes extensive cleaning and feature engineering to ensure consistency and reliability. Sentiment analysis is performed using VADER to extract polarity scores from textual reviews, enriching the dataset with derived emotional indicators.

A predictive machine learning pipeline is implemented to classify passenger satisfaction as a binary outcome (Satisfied / Dissatisfied). Machine learning models are developed and evaluated using accuracy, precision, recall, and F1-score. To ensure transparency, explainable AI techniques such as SHAP and LIME are applied to interpret feature importance and model behavior.

Beyond predictive analytics, the system transforms structured airline data into a Neo4j Knowledge Graph that explicitly models relationships between Passengers, Journeys, Flights, and Airports. This graph structure enables complex reasoning through Cypher queries, route analysis, delay aggregation, loyalty-based segmentation, and rule-based weighted satisfaction scoring. The knowledge graph serves as the factual grounding layer for the RAG system.

# Graph-RAG Architecture

The Graph-RAG pipeline processes user queries through multiple stages. First, the system performs intent classification and entity extraction to determine the user’s objective and identify relevant entities such as airports, flights, routes, or passenger attributes. Depending on the query type, the system retrieves relevant graph data using deterministic Cypher queries and embedding-based semantic similarity search.

Two retrieval strategies are implemented:

1. Baseline Retrieval – Structured Cypher queries retrieve precise graph matches using entity-based filtering and relationship traversal.

2. Embedding-Based Retrieval – Node embeddings are stored in Neo4j’s vector index to enable semantic similarity search, allowing more flexible and context-aware retrieval.

The results from both retrieval strategies are merged, deduplicated, and structured into a grounded context.

This context is then passed to multiple LLMs using a structured prompt format consisting of:

- Context (retrieved graph data),

- Persona (Airline Intelligence Assistant),

- Task (strict instruction to answer using only provided information).

The system compares at three LLMs, evaluating them using both quantitative metrics (accuracy, response time, token usage, cost) and qualitative analysis (answer relevance, correctness, and clarity).

# User Interface

A Streamlit-based user interface demonstrates the system in action. The UI allows users to:

- Submit natural language airline-related queries

- View the retrieved knowledge graph context before LLM processing

- See the executed Cypher queries

- Compare responses across different LLM models

- Switch between baseline retrieval, embedding-based retrieval, or hybrid mode

This design enhances transparency and makes the reasoning process interpretable and debuggable.

# System Strengths

This project demonstrates the integration of data engineering, machine learning, graph databases, and generative AI into a unified intelligent system. By grounding LLM responses in structured graph data, the system improves factual reliability while maintaining natural language flexibility.

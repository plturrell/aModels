# GloVe Embeddings Package

This package provides an OpenAI-compatible service for generating text embeddings using a pre-trained GloVe (Global Vectors for Word Representation) model. It allows the server to offer embedding capabilities locally without relying on an external service.

## Purpose

The primary purpose of this package is to expose a `/v1/embeddings` endpoint that functions just like the OpenAI Embeddings API. It takes a list of texts and returns a dense vector representation (embedding) for each one. This is useful for a variety of downstream tasks, such as semantic search, clustering, and text similarity calculations.

## How It Works

1.  **Model Loading**: The package is designed to load a pre-trained GloVe model into memory.
2.  **Text Processing**: When a request is received, the input text is tokenized (split into words), and the embedding for each word is retrieved from the GloVe model.
3.  **Sentence Embedding**: The word embeddings are then averaged to produce a single embedding for the entire input text.
4.  **Normalization**: The resulting sentence embedding is normalized to have a unit length, which is a standard practice for many similarity-based tasks.
5.  **Caching**: To improve performance, the package caches the generated embeddings in memory. If a request is received for a text that has already been processed, the cached embedding is returned immediately.
6.  **API Compatibility**: The request and response structs (`EmbeddingRequest`, `EmbeddingResponse`) are designed to be fully compatible with the OpenAI API, making it a seamless replacement.

## Key Files

*   `glove_localai.go`: Contains the `GloVeLocalAI` struct and all the logic for generating embeddings, caching, and computing similarities.

## Utility Functions

In addition to providing the core embedding generation service, this package also includes helper functions for common embedding-related tasks:

*   `ComputeSimilarity`: Calculates the cosine similarity between two texts.
*   `FindMostSimilar`: Finds the most similar text from a list of candidates for a given query.

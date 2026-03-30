# Reranking Guide

Complete guide to using Voyage AI rerankers with `mongodb-voyageai`.

## Table of Contents

1. [What is Reranking?](#what-is-reranking)
2. [Quick Start](#quick-start)
3. [Available Models](#available-models)
4. [Parameters](#parameters)
5. [Best Practices](#best-practices)
6. [Examples](#examples)
7. [Troubleshooting](#troubleshooting)

## What is Reranking?

A reranker refines the relevance ranking of candidate documents retrieved by an initial search system. Unlike embedding models that encode queries and documents separately, rerankers are **cross-encoders** that jointly process query-document pairs for more accurate relevance prediction.

### Why Use Reranking?

1. **Higher Accuracy**: Cross-encoders provide more precise relevance scores than embedding similarity
2. **Two-Stage Pipeline**: Fast embedding search (top 100) → Accurate reranking (top 10)
3. **Better Results**: Improves retrieval quality by 10-30% in most benchmarks

### Typical Pipeline

```
Query → [Embedding Search] → Top 100 candidates → [Reranker] → Top 10 results → LLM
```

## Quick Start

```rust
use mongodb_voyageai::{Client, Config, model};

#[tokio::main]
async fn main() -> Result<(), mongodb_voyageai::Error> {
    let client = Client::new(&Con
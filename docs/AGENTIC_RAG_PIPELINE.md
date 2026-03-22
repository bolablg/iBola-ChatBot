# Agentic RAG Pipeline

## Overview

The agentic RAG pipeline replaces the legacy `ConversationalRetrievalChain` with a **LangGraph state machine** that makes intelligent decisions about when to retrieve, when to grade, and when to rewrite queries.

## Graph Topology

```
guardrail ──┬── (score ≥ 60) ──▶ retrieve ──▶ grade_documents ──┬── (has relevant) ──▶ generate ──▶ END
            │                                                   │
            │                                                   ├── (no relevant, attempts < 2) ──▶ rewrite_query ──▶ retrieve
            │                                                   │
            │                                                   └── (no relevant, attempts ≥ 2) ──▶ generate (best-effort)
            │
            └── (score < 60) ──▶ out_of_scope ──▶ END
```

## Nodes

### 1. Guardrail Node
- **Purpose**: Score query relevance (0-100) and classify into category
- **LLM**: Gemini 2.5 Pro at **temperature 0.0** (deterministic)
- **Output**: `GuardrailScoring` Pydantic model with `score`, `category`, `reasoning`
- **Threshold**: score ≥ 60 → retrieve, score < 60 → out_of_scope
- **Fallback**: On LLM error, defaults to score=50 and routes to retrieve

### 2. Retrieve Node
- **Purpose**: Fetch documents using the specialized retriever for the detected category
- **Retrievers**: `ProfessionalRetriever`, `EducationRetriever`, `LearningRetriever`, `RedirectRetriever`
- **Tracks**: `retrieval_attempts` counter (max 2)

### 3. Grade Documents Node
- **Purpose**: Binary relevance grading for each retrieved document
- **LLM**: Gemini 2.5 Pro at **temperature 0.0** (deterministic)
- **Output**: `GradeDocuments` Pydantic model with `is_relevant`, `reasoning`
- **Fallback**: Content-length heuristic — documents > 50 chars are considered relevant

### 4. Rewrite Query Node
- **Purpose**: Reformulate vague or unsuccessful queries for better retrieval
- **LLM**: Gemini 2.5 Pro at **temperature 0.3** (controlled creativity)
- **Output**: `QueryRewrite` Pydantic model with `rewritten_query`
- **Fallback**: Appends "Bolaji BALOGOUN data science experience" to the original query

### 5. Generate Node
- **Purpose**: Generate the final answer from graded (relevant) documents
- **LLM**: Gemini 2.5 Pro at **temperature 0.7** (natural language)
- **Prompts**: Domain-specific per category (Professional, Education, Learning)
- **Context**: Only uses graded documents (irrelevant ones filtered out)

### 6. Out of Scope Node
- **Purpose**: Handle off-topic queries with polite redirect
- **Progressive behavior**: 1st redirect → polite suggestion, 2nd → contact options + end chat, 3rd+ → contact only

## Structured Outputs

All parsing-critical LLM calls use Pydantic models via `llm.with_structured_output(Model)`:

| Model | Temperature | Purpose |
|-------|-------------|---------|
| `GuardrailScoring` | 0.0 | Query scoring + classification |
| `GradeDocuments` | 0.0 | Binary document relevance |
| `QueryRewrite` | 0.3 | Query reformulation |

## State

`GraphState` is a dataclass passed through every node. Each node returns a partial `dict` update that gets merged into the state.

Key fields: `query`, `chat_history`, `category`, `guardrail_score`, `documents`, `graded_documents`, `answer`, `confidence`, `retrieval_attempts`, `reasoning_steps`.

## Service Wrapper

`AgenticRAGService` in `app/graph/service.py` wraps the compiled graph with:
- Session management (redirect counts, language, last agent)
- Google Chat alerts on high redirect counts
- Google Sheets redirect logging
- Contact request detection
- Error response fallback

## Files

| File | Purpose |
|------|---------|
| `app/graph/state.py` | GraphState dataclass + Pydantic structured output models |
| `app/graph/nodes.py` | All 6 node functions with fallbacks |
| `app/graph/workflow.py` | LangGraph StateGraph definition + conditional edges |
| `app/graph/service.py` | AgenticRAGService production wrapper |
| `app/graph/__init__.py` | Public exports |

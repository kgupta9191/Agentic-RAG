"""
conftest.py - pytest configuration and session-wide fixtures.

Mocks module-level LLM and vector-store initialisation in graph.py so that
the module can be imported without real OpenAI credentials or a FAISS index.
These patches are applied *before* graph.py is ever imported.
"""

import sys
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Build reusable mock objects that tests can introspect via fixtures below
# ---------------------------------------------------------------------------

# FAISS mock
mock_faiss_vector_db = MagicMock(name="faiss_vector_db")
mock_retriever = MagicMock(name="retriever")
mock_faiss_vector_db.as_retriever.return_value = mock_retriever

mock_faiss_cls = MagicMock(name="FAISS")
mock_faiss_cls.load_local.return_value = mock_faiss_vector_db
mock_faiss_cls.from_documents.return_value = mock_faiss_vector_db

# ChatOpenAI mock
mock_llm_instance = MagicMock(name="llm_instance")
mock_chat_openai_cls = MagicMock(name="ChatOpenAI", return_value=mock_llm_instance)

# OpenAIEmbeddings mock
mock_embeddings_instance = MagicMock(name="embeddings_instance")
mock_openai_embeddings_cls = MagicMock(
    name="OpenAIEmbeddings", return_value=mock_embeddings_instance
)

# ---------------------------------------------------------------------------
# Inject mocks into already-imported sub-modules so that `graph.py`'s
# top-level statements see the mocks when pytest first imports the test file.
# ---------------------------------------------------------------------------

import langchain_community.vectorstores as _lc_vs  # noqa: E402
import langchain_openai as _lc_oai  # noqa: E402

_lc_vs.FAISS = mock_faiss_cls
_lc_oai.ChatOpenAI = mock_chat_openai_cls
_lc_oai.OpenAIEmbeddings = mock_openai_embeddings_cls

# ---------------------------------------------------------------------------
# Now it is safe to import graph.py - module-level code will use the mocks
# ---------------------------------------------------------------------------

import graph  # noqa: E402, F401  (side-effect: sets graph.llm / graph.retriever)

# Ensure graph.retriever points to our controllable mock
graph.retriever = mock_retriever
graph.llm = mock_llm_instance

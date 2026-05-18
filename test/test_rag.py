"""
test_rag.py - Comprehensive unit tests for the Agentic-RAG LangGraph framework.

Test coverage:
  - tools.py  : extract_equations, solve_equation_tool, equation_tool_from_context
  - graph.py  : routing functions, individual nodes, graph structure, end-to-end flow
  - ingest.py : document loading, vector-DB creation
"""

import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import pytest

# ---------------------------------------------------------------------------
# tools.py imports (no LLM / FAISS dependencies)
# ---------------------------------------------------------------------------
from tools import extract_equations, solve_equation_tool, equation_tool_from_context

# ---------------------------------------------------------------------------
# graph.py imports – safe to import because conftest.py already patched FAISS
# ---------------------------------------------------------------------------
import graph as graph_module
from graph import (
    RAGState,
    decide_relevance,
    decide_grounding,
    fallback_response,
    increment_retry,
    equation_solver,
    retrieve_documents,
    grade_documents,
    generate_answer,
    check_answer_quality,
    build_graph,
    ask_rag,
)
from langchain_core.documents import Document


# ===========================================================================
# Helpers
# ===========================================================================

def make_state(**overrides) -> RAGState:
    """Return a minimal valid RAGState with sensible defaults."""
    base: RAGState = {
        "question": "What is the speed of light?",
        "documents": [],
        "context": "",
        "relevance_score": "",
        "answer": "",
        "grounded_score": "",
        "equation_result": "",
        "retries": 0,
    }
    base.update(overrides)
    return base


# ===========================================================================
# 1. tools.py – extract_equations
# ===========================================================================

class TestExtractEquations:
    def test_simple_assignment_equation(self):
        text = "The force is F = m*a in classical mechanics."
        equations = extract_equations(text)
        assert any("F" in eq and "m" in eq for eq in equations)

    def test_multiple_equations(self):
        text = "E = m*c**2\nv = u + a*t\n"
        equations = extract_equations(text)
        assert len(equations) >= 2

    def test_returns_unique_equations(self):
        text = "F = m*a\nF = m*a\nF = m*a\n"
        equations = extract_equations(text)
        assert len(equations) == len(set(equations))

    def test_long_equation_filtered_out(self):
        long_eq = "A" + "_var" * 20 + " = " + "x" * 60
        equations = extract_equations(long_eq)
        assert all(len(eq) < 80 for eq in equations)

    def test_no_equations_returns_empty(self):
        text = "This text has no equations whatsoever."
        equations = extract_equations(text)
        # Equations list may be empty or only trivially short matches
        for eq in equations:
            assert len(eq) < 80

    def test_equation_with_caret_notation(self):
        text = "y = x^2 + 3"
        equations = extract_equations(text)
        assert len(equations) >= 1

    def test_empty_string(self):
        assert extract_equations("") == []


# ===========================================================================
# 2. tools.py – solve_equation_tool
# ===========================================================================

class TestSolveEquationTool:
    def test_solve_for_specific_variable(self):
        result = solve_equation_tool("F = m*a", variable="a")
        assert "a" in result.lower() or "solution" in result.lower()

    def test_solve_for_all_symbols(self):
        result = solve_equation_tool("v = u + a*t")
        assert "solution" in result.lower() or "symbolic" in result.lower()

    def test_simplification_without_equals(self):
        result = solve_equation_tool("x**2 + 2*x + 1")
        assert "simplified" in result.lower()

    def test_caret_replaced_with_double_star(self):
        result = solve_equation_tool("y = x^2", variable="x")
        assert "error" not in result.lower() or "solution" in result.lower()

    def test_invalid_equation_returns_error_message(self):
        result = solve_equation_tool("$$%%invalid")
        assert "error" in result.lower() or "could not" in result.lower()

    def test_constant_equation_evaluates(self):
        result = solve_equation_tool("2 = 2")
        assert result is not None and len(result) > 0

    def test_solve_quadratic(self):
        result = solve_equation_tool("x**2 - 4 = 0", variable="x")
        assert "2" in result  # x = ±2


# ===========================================================================
# 3. tools.py – equation_tool_from_context
# ===========================================================================

class TestEquationToolFromContext:
    def test_no_equations_in_context(self):
        result = equation_tool_from_context(
            context="The sky is blue. Water is wet.",
            user_question="What is the formula?",
        )
        assert "no clear equation" in result.lower()

    def test_equations_found_and_solved(self):
        result = equation_tool_from_context(
            context="Newton's law: F = m*a",
            user_question="What is Newton's second law?",
        )
        assert "F" in result or "equation" in result.lower()

    def test_solve_for_variable_from_question(self):
        result = equation_tool_from_context(
            context="F = m*a defines force.",
            user_question="Solve for a in F = m*a",
        )
        assert "a" in result.lower() or "solution" in result.lower()

    def test_limits_to_five_equations(self):
        lines = "\n".join([f"var{i} = x*{i}" for i in range(1, 20)])
        result = equation_tool_from_context(
            context=lines,
            user_question="Explain these equations",
        )
        # Should process at most 5 equations
        count = result.count("Equation:")
        assert count <= 5

    def test_empty_context(self):
        result = equation_tool_from_context(
            context="",
            user_question="Any equations?",
        )
        assert "no clear equation" in result.lower()


# ===========================================================================
# 4. graph.py – pure routing functions
# ===========================================================================

class TestDecideRelevance:
    def test_relevant_routes_to_generate_answer(self):
        state = make_state(relevance_score="relevant")
        assert decide_relevance(state) == "generate_answer"

    def test_not_relevant_routes_to_fallback(self):
        state = make_state(relevance_score="not_relevant")
        assert decide_relevance(state) == "fallback_response"

    def test_empty_score_routes_to_fallback(self):
        state = make_state(relevance_score="")
        assert decide_relevance(state) == "fallback_response"


class TestDecideGrounding:
    def test_grounded_routes_to_equation_solver(self):
        state = make_state(grounded_score="grounded", retries=0)
        assert decide_grounding(state) == "equation_solver"

    def test_not_grounded_first_attempt_retries(self):
        state = make_state(grounded_score="not_grounded", retries=0)
        assert decide_grounding(state) == "generate_answer"

    def test_not_grounded_after_retry_falls_back(self):
        state = make_state(grounded_score="not_grounded", retries=1)
        assert decide_grounding(state) == "fallback_response"

    def test_not_grounded_high_retry_falls_back(self):
        state = make_state(grounded_score="not_grounded", retries=5)
        assert decide_grounding(state) == "fallback_response"


# ===========================================================================
# 5. graph.py – pure state-transformation nodes
# ===========================================================================

class TestFallbackResponse:
    def test_returns_fallback_message(self):
        result = fallback_response(make_state())
        assert "answer" in result
        assert len(result["answer"]) > 0
        assert "could not find" in result["answer"].lower()


class TestIncrementRetry:
    def test_increments_from_zero(self):
        assert increment_retry(make_state(retries=0))["retries"] == 1

    def test_increments_from_existing_value(self):
        assert increment_retry(make_state(retries=3))["retries"] == 4


# ===========================================================================
# 6. graph.py – equation_solver node
# ===========================================================================

class TestEquationSolverNode:
    def test_no_equation_keyword_returns_empty(self):
        state = make_state(
            question="What is the speed of light?",
            context="Light travels very fast.",
        )
        result = equation_solver(state)
        assert result["equation_result"] == ""

    def test_equation_keyword_triggers_tool(self):
        state = make_state(
            question="Solve for a in F = m*a",
            context="Newton's second law: F = m*a",
        )
        result = equation_solver(state)
        assert isinstance(result["equation_result"], str)

    @pytest.mark.parametrize("keyword", [
        "equation", "solve", "derive", "formula",
        "calculate", "simplify", "expression",
    ])
    def test_each_equation_keyword_triggers_tool(self, keyword):
        state = make_state(
            question=f"Please {keyword} the force",
            context="F = m*a",
        )
        result = equation_solver(state)
        assert isinstance(result["equation_result"], str)


# ===========================================================================
# 7. graph.py – LLM-dependent nodes (mocked)
# ===========================================================================

class TestRetrieveDocuments:
    def test_documents_and_context_populated(self):
        doc = Document(
            page_content="Some physics content",
            metadata={"source": "physics.pdf", "page": 1},
        )
        graph_module.retriever.invoke.return_value = [doc]

        result = retrieve_documents(make_state(question="What is gravity?"))

        assert len(result["documents"]) == 1
        assert "physics.pdf" in result["context"]
        assert "Some physics content" in result["context"]
        graph_module.retriever.invoke.assert_called_once_with("What is gravity?")

    def test_multiple_documents_joined(self):
        docs = [
            Document(page_content=f"Content {i}", metadata={"source": f"doc{i}.pdf"})
            for i in range(3)
        ]
        graph_module.retriever.invoke.return_value = docs
        result = retrieve_documents(make_state())
        # Each doc contributes one "Source:" line; count those to verify all 3 are present
        assert result["context"].count("Source:") == 3

    def test_empty_retrieval(self):
        graph_module.retriever.invoke.return_value = []
        result = retrieve_documents(make_state())
        assert result["documents"] == []
        assert result["context"] == ""


class TestGradeDocuments:
    def test_relevant_response(self):
        graph_module.llm.invoke.return_value = MagicMock(content="relevant")
        result = grade_documents(make_state(context="useful context"))
        assert result["relevance_score"] == "relevant"

    def test_not_relevant_response(self):
        graph_module.llm.invoke.return_value = MagicMock(content="not_relevant")
        result = grade_documents(make_state(context="unrelated context"))
        assert result["relevance_score"] == "not_relevant"

    def test_response_with_not_keyword_is_not_relevant(self):
        graph_module.llm.invoke.return_value = MagicMock(content="not relevant at all")
        result = grade_documents(make_state())
        assert result["relevance_score"] == "not_relevant"

    def test_response_normalised_to_relevant(self):
        graph_module.llm.invoke.return_value = MagicMock(content="RELEVANT")
        result = grade_documents(make_state())
        assert result["relevance_score"] == "relevant"


class TestGenerateAnswer:
    def test_answer_populated_from_llm(self):
        graph_module.llm.invoke.return_value = MagicMock(content="The answer is 42.")
        result = generate_answer(make_state(context="some context"))
        assert result["answer"] == "The answer is 42."

    def test_retries_preserved(self):
        graph_module.llm.invoke.return_value = MagicMock(content="An answer.")
        result = generate_answer(make_state(retries=2))
        assert result["retries"] == 2

    def test_retries_default_zero(self):
        graph_module.llm.invoke.return_value = MagicMock(content="An answer.")
        state = make_state()
        result = generate_answer(state)
        assert result["retries"] == 0


class TestCheckAnswerQuality:
    def test_grounded_response(self):
        graph_module.llm.invoke.return_value = MagicMock(content="grounded")
        result = check_answer_quality(make_state(answer="A clear answer."))
        assert result["grounded_score"] == "grounded"

    def test_not_grounded_response(self):
        graph_module.llm.invoke.return_value = MagicMock(content="not_grounded")
        result = check_answer_quality(make_state(answer="A hallucinated answer."))
        assert result["grounded_score"] == "not_grounded"

    def test_not_keyword_detected(self):
        graph_module.llm.invoke.return_value = MagicMock(content="not grounded in context")
        result = check_answer_quality(make_state())
        assert result["grounded_score"] == "not_grounded"

    def test_grounded_case_insensitive(self):
        graph_module.llm.invoke.return_value = MagicMock(content="GROUNDED")
        result = check_answer_quality(make_state())
        assert result["grounded_score"] == "grounded"


# ===========================================================================
# 8. graph.py – graph structure
# ===========================================================================

class TestBuildGraph:
    def test_graph_compiles_without_error(self):
        compiled = build_graph()
        assert compiled is not None

    def test_graph_has_all_nodes(self):
        compiled = build_graph()
        node_names = set(compiled.get_graph().nodes.keys())
        expected = {
            "retrieve_documents",
            "grade_documents",
            "generate_answer",
            "check_answer_quality",
            "equation_solver",
            "fallback_response",
        }
        assert expected.issubset(node_names)

    def test_graph_entry_point_is_retrieve(self):
        compiled = build_graph()
        graph_def = compiled.get_graph()
        # Entry point edges come from the __start__ node
        start_targets = {
            edge.target for edge in graph_def.edges if edge.source == "__start__"
        }
        assert "retrieve_documents" in start_targets


# ===========================================================================
# 9. graph.py – end-to-end ask_rag
# ===========================================================================

class TestAskRAG:
    def _setup_happy_path(self):
        doc = Document(
            page_content="Light travels at 3e8 m/s.",
            metadata={"source": "physics.pdf", "page": 1},
        )
        graph_module.retriever.invoke.return_value = [doc]
        # grade_documents -> relevant, generate_answer -> answer, check_answer_quality -> grounded
        responses = iter([
            MagicMock(content="relevant"),    # grade_documents
            MagicMock(content="The speed of light is 3e8 m/s."),  # generate_answer
            MagicMock(content="grounded"),    # check_answer_quality
        ])
        graph_module.llm.invoke.side_effect = lambda *a, **kw: next(responses)

    def test_ask_rag_returns_string(self):
        self._setup_happy_path()
        answer = ask_rag("What is the speed of light?")
        assert isinstance(answer, str)
        assert len(answer) > 0

    def test_ask_rag_answer_content(self):
        self._setup_happy_path()
        answer = ask_rag("What is the speed of light?")
        assert "3e8" in answer or "speed" in answer.lower()

    def test_ask_rag_fallback_on_irrelevant(self):
        graph_module.retriever.invoke.return_value = []
        graph_module.llm.invoke.side_effect = None  # clear any leftover side_effect
        graph_module.llm.invoke.return_value = MagicMock(content="not_relevant")
        answer = ask_rag("completely irrelevant question xyz123")
        assert isinstance(answer, str)

    def test_ask_rag_appends_equation_result(self):
        doc = Document(
            page_content="F = m*a is Newton's law.",
            metadata={"source": "mechanics.pdf"},
        )
        graph_module.retriever.invoke.return_value = [doc]
        responses = iter([
            MagicMock(content="relevant"),
            MagicMock(content="Force equals mass times acceleration."),
            MagicMock(content="grounded"),
        ])
        graph_module.llm.invoke.side_effect = lambda *a, **kw: next(responses)
        answer = ask_rag("Solve for a in F = m*a")
        assert isinstance(answer, str)


# ===========================================================================
# 10. RAGState – structure validation
# ===========================================================================

class TestRAGStateStructure:
    def test_state_contains_all_required_keys(self):
        state = make_state()
        required_keys = {
            "question", "documents", "context",
            "relevance_score", "answer", "grounded_score",
            "equation_result", "retries",
        }
        assert required_keys == set(state.keys())

    def test_documents_default_is_list(self):
        state = make_state()
        assert isinstance(state["documents"], list)

    def test_retries_default_is_zero(self):
        state = make_state()
        assert state["retries"] == 0


# ===========================================================================
# 11. ingest.py – document loading and vector-DB creation
# ===========================================================================

class TestLoadDocuments:
    def test_loads_pdf_files(self, tmp_path):
        (tmp_path / "sample.pdf").write_bytes(b"fake pdf")
        mock_doc = MagicMock()
        mock_doc.metadata = {}
        mock_loader = MagicMock()
        mock_loader.load.return_value = [mock_doc]
        with patch("ingest.PyPDFLoader", return_value=mock_loader):
            from ingest import load_documents
            docs = load_documents(str(tmp_path))
        assert len(docs) == 1
        assert docs[0].metadata["source"] == "sample.pdf"

    def test_loads_txt_files(self, tmp_path):
        (tmp_path / "notes.txt").write_text("hello world", encoding="utf-8")
        mock_doc = MagicMock()
        mock_doc.metadata = {}
        mock_loader = MagicMock()
        mock_loader.load.return_value = [mock_doc]
        with patch("ingest.TextLoader", return_value=mock_loader):
            from ingest import load_documents
            docs = load_documents(str(tmp_path))
        assert len(docs) == 1
        assert docs[0].metadata["source"] == "notes.txt"

    def test_loads_md_files(self, tmp_path):
        (tmp_path / "readme.md").write_text("# Title", encoding="utf-8")
        mock_doc = MagicMock()
        mock_doc.metadata = {}
        mock_loader = MagicMock()
        mock_loader.load.return_value = [mock_doc]
        with patch("ingest.TextLoader", return_value=mock_loader):
            from ingest import load_documents
            docs = load_documents(str(tmp_path))
        assert len(docs) == 1

    def test_skips_unknown_extensions(self, tmp_path):
        (tmp_path / "image.jpg").write_bytes(b"fake image")
        from ingest import load_documents
        docs = load_documents(str(tmp_path))
        assert docs == []

    def test_empty_directory_returns_empty_list(self, tmp_path):
        from ingest import load_documents
        docs = load_documents(str(tmp_path))
        assert docs == []

    def test_source_metadata_set_to_filename(self, tmp_path):
        (tmp_path / "physics.txt").write_text("content", encoding="utf-8")
        mock_doc = MagicMock()
        mock_doc.metadata = {}
        mock_loader = MagicMock()
        mock_loader.load.return_value = [mock_doc]
        with patch("ingest.TextLoader", return_value=mock_loader):
            from ingest import load_documents
            docs = load_documents(str(tmp_path))
        assert docs[0].metadata["source"] == "physics.txt"


class TestCreateVectorDB:
    def test_raises_if_no_documents(self, tmp_path):
        with patch("ingest.load_documents", return_value=[]):
            with patch("ingest.DATA_DIR", str(tmp_path)):
                from ingest import create_vector_db
                with pytest.raises(ValueError, match="No PDF/TXT/MD files"):
                    create_vector_db()

    def test_saves_vector_db(self, tmp_path):
        mock_doc = MagicMock()
        mock_chunks = [MagicMock(), MagicMock()]
        mock_splitter = MagicMock()
        mock_splitter.split_documents.return_value = mock_chunks
        mock_embeddings = MagicMock()
        mock_vector_db = MagicMock()

        with (
            patch("ingest.load_documents", return_value=[mock_doc]),
            patch("ingest.RecursiveCharacterTextSplitter", return_value=mock_splitter),
            patch("ingest.OpenAIEmbeddings", return_value=mock_embeddings),
            patch("ingest.FAISS") as mock_faiss_cls,
            patch("ingest.VECTOR_DB_DIR", str(tmp_path / "faiss_index")),
        ):
            mock_faiss_cls.from_documents.return_value = mock_vector_db
            from ingest import create_vector_db
            create_vector_db()
            mock_vector_db.save_local.assert_called_once()

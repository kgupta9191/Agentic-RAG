from typing import TypedDict, List, Literal
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, END

from tools import equation_tool_from_context

load_dotenv()

VECTOR_DB_DIR = "vector_db/faiss_index"


class RAGState(TypedDict):
    question: str
    documents: List[Document]
    context: str
    relevance_score: str
    answer: str
    grounded_score: str
    equation_result: str
    retries: int


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector_db = FAISS.load_local(
    VECTOR_DB_DIR,
    embeddings,
    allow_dangerous_deserialization=True,
)

retriever = vector_db.as_retriever(
    search_kwargs={"k": 5}
)


def retrieve_documents(state: RAGState):
    question = state["question"]

    docs = retriever.invoke(question)

    context = "\n\n".join(
        [
            f"Source: {doc.metadata.get('source', 'unknown')}\n"
            f"Page: {doc.metadata.get('page', 'unknown')}\n"
            f"Content:\n{doc.page_content}"
            for doc in docs
        ]
    )

    return {
        "documents": docs,
        "context": context,
    }


def grade_documents(state: RAGState):
    prompt = f"""
You are a document relevance grader.

Question:
{state["question"]}

Retrieved context:
{state["context"]}

Decide if the retrieved context is useful for answering the question.

Return only one word:
relevant
or
not_relevant
"""

    response = llm.invoke(prompt)
    score = response.content.strip().lower()

    if "not" in score:
        score = "not_relevant"
    else:
        score = "relevant"

    return {"relevance_score": score}


def decide_relevance(state: RAGState) -> Literal["generate_answer", "fallback_response"]:
    if state["relevance_score"] == "relevant":
        return "generate_answer"
    return "fallback_response"


def generate_answer(state: RAGState):
    prompt = f"""
You are a careful RAG assistant.

Use only the retrieved context to answer the question.
If the answer is not clearly supported by the context, say that the documents do not contain enough information.

Question:
{state["question"]}

Context:
{state["context"]}

Answer clearly and scientifically.
Mention the source filenames/pages when useful.
"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "retries": state.get("retries", 0),
    }


def check_answer_quality(state: RAGState):
    prompt = f"""
You are an answer grounding checker.

Question:
{state["question"]}

Context:
{state["context"]}

Answer:
{state["answer"]}

Check if the answer is fully supported by the context.

Return only one word:
grounded
or
not_grounded
"""

    response = llm.invoke(prompt)
    score = response.content.strip().lower()

    if "not" in score:
        score = "not_grounded"
    else:
        score = "grounded"

    return {"grounded_score": score}


def decide_grounding(state: RAGState) -> Literal["equation_solver", "generate_answer", "fallback_response"]:
    retries = state.get("retries", 0)

    if state["grounded_score"] == "grounded":
        return "equation_solver"

    if retries < 1:
        return "generate_answer"

    return "fallback_response"


def equation_solver(state: RAGState):
    question = state["question"].lower()

    equation_keywords = [
        "equation",
        "solve",
        "derive",
        "formula",
        "calculate",
        "simplify",
        "expression",
    ]

    if not any(word in question for word in equation_keywords):
        return {"equation_result": ""}

    result = equation_tool_from_context(
        context=state["context"],
        user_question=state["question"],
    )

    return {"equation_result": result}


def fallback_response(state: RAGState):
    return {
        "answer": (
            "I could not find enough relevant or grounded information in the retrieved documents "
            "to answer this reliably. Try adding more documents, improving the question, or rebuilding "
            "the vector database with better chunking."
        )
    }


def increment_retry(state: RAGState):
    retries = state.get("retries", 0)
    return {"retries": retries + 1}


def build_graph():
    graph = StateGraph(RAGState)

    graph.add_node("retrieve_documents", retrieve_documents)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("check_answer_quality", check_answer_quality)
    graph.add_node("equation_solver", equation_solver)
    graph.add_node("fallback_response", fallback_response)

    graph.set_entry_point("retrieve_documents")

    graph.add_edge("retrieve_documents", "grade_documents")

    graph.add_conditional_edges(
        "grade_documents",
        decide_relevance,
        {
            "generate_answer": "generate_answer",
            "fallback_response": "fallback_response",
        },
    )

    graph.add_edge("generate_answer", "check_answer_quality")

    graph.add_conditional_edges(
        "check_answer_quality",
        decide_grounding,
        {
            "equation_solver": "equation_solver",
            "generate_answer": "generate_answer",
            "fallback_response": "fallback_response",
        },
    )

    graph.add_edge("equation_solver", END)
    graph.add_edge("fallback_response", END)

    return graph.compile()


rag_graph = build_graph()


def ask_rag(question: str):
    result = rag_graph.invoke(
        {
            "question": question,
            "documents": [],
            "context": "",
            "relevance_score": "",
            "answer": "",
            "grounded_score": "",
            "equation_result": "",
            "retries": 0,
        }
    )

    final_answer = result.get("answer", "")

    equation_result = result.get("equation_result", "")
    if equation_result:
        final_answer += "\n\n---\n\nEquation tool result:\n\n" + equation_result

    return final_answer

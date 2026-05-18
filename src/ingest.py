import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

DATA_DIR = "data"
VECTOR_DB_DIR = "vector_db/faiss_index"


def load_documents(data_dir: str):
    documents = []

    for file_path in Path(data_dir).glob("*"):
        if file_path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(file_path))
            docs = loader.load()

        elif file_path.suffix.lower() in [".txt", ".md"]:
            loader = TextLoader(str(file_path), encoding="utf-8")
            docs = loader.load()

        else:
            continue

        for doc in docs:
            doc.metadata["source"] = file_path.name

        documents.extend(docs)

    return documents


def create_vector_db():
    print("Loading documents...")
    documents = load_documents(DATA_DIR)

    if not documents:
        raise ValueError("No PDF/TXT/MD files found in data/ folder.")

    print(f"Loaded {len(documents)} raw documents/pages.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vector_db = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    os.makedirs(os.path.dirname(VECTOR_DB_DIR), exist_ok=True)
    vector_db.save_local(VECTOR_DB_DIR)

    print(f"Vector database saved to: {VECTOR_DB_DIR}")


if __name__ == "__main__":
    create_vector_db()

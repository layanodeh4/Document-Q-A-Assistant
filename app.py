import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer


# Load environment variables
load_dotenv()

# Create OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Folder containing our documents
DOCS_FOLDER = Path("docs")

# Number of chunks to retrieve
TOP_K = 4


class QAResponse(BaseModel):
    answer: str
    citations: list[int] = Field(default_factory=list)


def load_documents():
    """
    Load all Markdown and text files from the docs folder.
    """

    documents = []

    for file_path in DOCS_FOLDER.iterdir():

        if file_path.suffix.lower() in [".md", ".txt"]:

            text = file_path.read_text(encoding="utf-8")

            documents.append({
                "filename": file_path.name,
                "text": text
            })

    return documents


def chunk_text(text, chunk_size=80):
    """
    Split text into smaller chunks.
    """

    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


def create_chunks(documents):
    """
    Create chunks while keeping the source filename.
    """

    all_chunks = []

    chunk_id = 1

    for document in documents:

        chunks = chunk_text(document["text"])

        for chunk in chunks:

            all_chunks.append({
                "id": chunk_id,
                "filename": document["filename"],
                "text": chunk
            })

            chunk_id += 1

    return all_chunks


def create_embeddings(chunks):
    """
    Create embeddings for all document chunks.
    """

    texts = [chunk["text"] for chunk in chunks]

    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True
    )

    return np.array(embeddings)


def retrieve_chunks(question, chunks, chunk_embeddings):
    """
    Retrieve the top 4 most relevant chunks.
    """

    question_embedding = embedding_model.encode(
        question,
        normalize_embeddings=True
    )

    # Cosine similarity because embeddings are normalized
    scores = np.dot(chunk_embeddings, question_embedding)

    top_indices = np.argsort(scores)[::-1][:TOP_K]

    retrieved = []

    for index in top_indices:

        retrieved.append({
            "id": chunks[index]["id"],
            "filename": chunks[index]["filename"],
            "text": chunks[index]["text"],
            "score": float(scores[index])
        })

    return retrieved


def build_context(retrieved_chunks):
    """
    Build the context that will be sent to the LLM.
    """

    context_parts = []

    for chunk in retrieved_chunks:

        context_parts.append(
            f"[{chunk['id']}] "
            f"Source: {chunk['filename']}\n"
            f"{chunk['text']}"
        )

    return "\n\n".join(context_parts)


def ask_llm(question, retrieved_chunks):
    """
    Ask the language model to answer only from retrieved context.
    """

    context = build_context(retrieved_chunks)

    prompt = f"""
You are a document question-answering assistant.

Answer the user's question using ONLY the information in the
retrieved document passages below.

If the answer is not contained in the passages, say:
"I don't know."

Do not use outside knowledge.
Do not invent information.

You must return valid JSON with exactly this structure:

{{
    "answer": "your answer",
    "citations": [1, 2]
}}

The citations must contain the IDs of the passages used to answer
the question.

If the answer is "I don't know.", return an empty citations list.

Retrieved passages:

{context}

User question:

{question}
"""

    response = client.responses.create(
        model="gpt-5.6-luna",
        input=prompt
    )

    return response.output_text


def validate_response(raw_response):
    """
    Validate the LLM response using Pydantic.
    """

    import json

    data = json.loads(raw_response)

    validated = QAResponse.model_validate(data)

    return validated


def answer_question(question, chunks, chunk_embeddings):
    """
    Complete question-answering pipeline.
    """

    retrieved = retrieve_chunks(
        question,
        chunks,
        chunk_embeddings
    )

    print("\n" + "=" * 60)
    print("QUESTION")
    print(question)

    print("\nRETRIEVED CHUNKS")

    for chunk in retrieved:

        print(
            f"\n[{chunk['id']}] "
            f"{chunk['filename']} "
            f"(score={chunk['score']:.3f})"
        )

        print(chunk["text"])

    raw_response = ask_llm(
        question,
        retrieved
    )

    try:

        validated = validate_response(raw_response)

        print("\nFINAL ANSWER")
        print(validated.answer)

        print("\nCITATIONS")
        print(validated.citations)

        print("\nVALIDATION")
        print("Passed")

        return validated

    except Exception as error:

        print("\nVALIDATION")
        print("Failed")

        print("Error:", error)

        return None


def main():

    print("Loading documents...")

    documents = load_documents()

    print(f"Loaded {len(documents)} documents.")

    print("\nCreating chunks...")

    chunks = create_chunks(documents)

    print(f"Created {len(chunks)} chunks.")

    print("\nCreating embeddings...")

    chunk_embeddings = create_embeddings(chunks)

    print("Embeddings created successfully.")

    print("\nDocument Q&A Assistant")
    print("Type 'exit' to stop.")

    while True:

        question = input("\nAsk a question: ")

        if question.lower() == "exit":
            break

        answer_question(
            question,
            chunks,
            chunk_embeddings
        )


if __name__ == "__main__":
    main()
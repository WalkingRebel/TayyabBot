from pathlib import Path
import pickle

import faiss
import numpy as np
from fastembed import TextEmbedding

from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    VECTORSTORE_DIR,
    INDEX_FILE,
    METADATA_FILE,
)


DATASET_DIR = Path("dataset")
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def clean_text(text):
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    lines = [
        line.strip()
        for line in text.split("\n")
        if line.strip()
    ]

    return "\n".join(lines)


def create_chunks(text, source):
    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(
            start + CHUNK_SIZE,
            text_length
        )

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(
                {
                    "text": chunk,
                    "source": source
                }
            )

        if end >= text_length:
            break

        start = end - CHUNK_OVERLAP

    return chunks


def load_and_process_documents():
    all_chunks = []

    files = sorted(DATASET_DIR.glob("*.txt"))

    if not files:
        raise FileNotFoundError(
            "No .txt files were found inside the dataset folder."
        )

    print(f"Found {len(files)} dataset files.")

    for file_path in files:
        print(f"Loading: {file_path.name}")

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as file:
            text = file.read()

        cleaned_text = clean_text(text)

        chunks = create_chunks(
            cleaned_text,
            file_path.name
        )

        all_chunks.extend(chunks)

    print(
        f"Created {len(all_chunks)} chunks "
        "from the personal dataset."
    )

    return all_chunks


def generate_embeddings(texts):
    print("Loading local embedding model...")

    model = TextEmbedding(
        model_name=EMBEDDING_MODEL
    )

    print("Generating embeddings...")

    embeddings = list(
        model.embed(
            [f"passage: {text}" for text in texts]
        )
    )

    return np.array(
        embeddings,
        dtype="float32"
    )


def create_vectorstore(chunks):
    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = generate_embeddings(
        texts
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(embeddings)

    Path(
        VECTORSTORE_DIR
    ).mkdir(
        parents=True,
        exist_ok=True
    )

    faiss.write_index(
        index,
        INDEX_FILE
    )

    with open(
        METADATA_FILE,
        "wb"
    ) as file:
        pickle.dump(
            chunks,
            file
        )

    print()
    print(
        "Vector database created successfully."
    )
    print(
        f"Total vectors: {index.ntotal}"
    )
    print(
        f"Embedding dimension: {dimension}"
    )
    print(
        f"Index: {INDEX_FILE}"
    )
    print(
        f"Metadata: {METADATA_FILE}"
    )


def main():
    print("=" * 60)
    print("TayyabBot - Dataset Ingestion")
    print("=" * 60)

    chunks = load_and_process_documents()

    create_vectorstore(chunks)

    print("=" * 60)
    print("Ingestion completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
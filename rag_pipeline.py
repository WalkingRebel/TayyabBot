import pickle

import faiss
import numpy as np
from fastembed import TextEmbedding
from groq import Groq

from config import (
    GROQ_API_KEY,
    LLM_MODEL,
    INDEX_FILE,
    METADATA_FILE,
    TOP_K,
)


EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


class TayyabRAG:

    def __init__(self):

        if not GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is missing. "
                "Add it to your .env file."
            )

        self.client = Groq(
            api_key=GROQ_API_KEY
        )

        self.embedding_model = TextEmbedding(
            model_name=EMBEDDING_MODEL
        )

        self.index = faiss.read_index(
            INDEX_FILE
        )

        with open(
            METADATA_FILE,
            "rb"
        ) as file:
            self.metadata = pickle.load(file)

        self.history = []


    def embed_query(self, query):

        embedding = list(
            self.embedding_model.embed(
                [f"query: {query}"]
            )
        )[0]

        return np.array(
            [embedding],
            dtype="float32"
        )


    def retrieve(self, query):

        query_embedding = self.embed_query(
            query
        )

        distances, indices = self.index.search(
            query_embedding,
            TOP_K
        )

        retrieved_chunks = []

        for distance, index in zip(
            distances[0],
            indices[0]
        ):

            if index < 0:
                continue

            chunk = self.metadata[index].copy()

            chunk["distance"] = float(
                distance
            )

            retrieved_chunks.append(
                chunk
            )

        return retrieved_chunks


    def build_context(
        self,
        retrieved_chunks
    ):

        context_parts = []

        for i, chunk in enumerate(
            retrieved_chunks,
            start=1
        ):

            context_parts.append(
                f"[Source {i}: {chunk['source']}]\n"
                f"{chunk['text']}"
            )

        return "\n\n".join(
            context_parts
        )


    def build_history(self):

        if not self.history:
            return "No previous conversation."

        recent_history = self.history[-6:]

        history_text = []

        for message in recent_history:

            role = message["role"]
            content = message["content"]

            history_text.append(
                f"{role}: {content}"
            )

        return "\n".join(
            history_text
        )


    def generate_response(
        self,
        query,
        retrieved_chunks
    ):

        context = self.build_context(
            retrieved_chunks
        )

        history = self.build_history()

        system_prompt = """
You are TayyabBot, a personalized
Retrieval-Augmented Generation chatbot
created for Muhammad Tayyab.

Your purpose is to answer questions about
Tayyab using the personal dataset provided
through retrieval.

IMPORTANT RULES:

1. Use the retrieved context as your
   primary source of information.

2. Do not invent personal information.

3. If the retrieved context does not contain
   enough information to answer the question,
   clearly say that the information is not
   available in Tayyab's personal dataset.

4. Do not present assumptions as facts.

5. Keep answers clear, natural, and relevant.

6. Use conversation history when it helps
   understand the current question.

7. When appropriate, mention the dataset
   source supporting the answer.

8. You are TayyabBot, not Muhammad Tayyab.
   Never claim to personally be Tayyab.

9. Stay grounded in the retrieved context.

10. If the user asks a question unrelated
    to Tayyab's personal information, explain
    that TayyabBot is designed to answer
    questions based on Tayyab's personal
    dataset.
"""

        user_prompt = f"""
Retrieved Context:

{context}

Conversation History:

{history}

Current User Question:

{query}

Answer the user's question using the
retrieved context and conversation history.
"""

        response = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            temperature=0.2
        )

        answer = response.choices[0].message.content

        self.history.append(
            {
                "role": "user",
                "content": query
            }
        )

        self.history.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        return answer


    def ask(self, query):

        retrieved_chunks = self.retrieve(
            query
        )

        answer = self.generate_response(
            query,
            retrieved_chunks
        )

        sources = [
            chunk["source"]
            for chunk in retrieved_chunks
        ]

        return answer, sources
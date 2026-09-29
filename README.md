# TayyabBot 🤖

TayyabBot is a personalized Retrieval-Augmented Generation (RAG) chatbot developed as a Natural Language Processing semester project at the University of Management and Technology (UMT), Lahore.

The chatbot answers questions about Muhammad Tayyab using a custom personal dataset. It retrieves relevant information from the dataset using semantic embeddings and FAISS, then uses an LLM to generate a grounded response.

---

## Project Overview

TayyabBot demonstrates an end-to-end RAG pipeline:

Personal Dataset  
↓  
Text Preprocessing  
↓  
Text Chunking  
↓  
Local Embeddings using FastEmbed  
↓  
FAISS Vector Database  
↓  
Similarity Retrieval  
↓  
Retrieved Context  
↓  
Prompt Engineering + Conversation History  
↓  
Groq LLM  
↓  
TayyabBot Response

The system is designed to reduce hallucination by grounding responses in the retrieved personal dataset.

---

## Project Objectives

The main objectives of TayyabBot are:

- Build a personalized NLP chatbot.
- Use a custom personal dataset instead of relying only on generic datasets.
- Preprocess and index personal information.
- Generate semantic embeddings locally.
- Store embeddings in a FAISS vector database.
- Retrieve relevant information for each user query.
- Generate responses using an LLM.
- Apply prompt engineering to keep responses grounded.
- Maintain conversation history.
- Provide an interactive Streamlit chatbot interface.
- Deploy the chatbot for online access.

---

## Technologies Used

### Programming Language

- Python

### RAG Components

- FastEmbed
- FAISS
- NumPy
- Groq API

### User Interface

- Streamlit

### Environment Management

- uv
- Python virtual environment

### Configuration

- python-dotenv

---

## Personal Dataset

TayyabBot uses a custom personal dataset created specifically for this project.

The dataset contains information about:

- Personal profile
- Education
- Skills
- Courses
- Certifications
- Experience
- Projects
- Goals
- Resume

Dataset files are stored inside:

```text
dataset/
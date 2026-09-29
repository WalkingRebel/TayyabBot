# TayyabBot

TayyabBot is a personalized Retrieval-Augmented Generation (RAG) chatbot developed as a semester project for the Natural Language Processing course at the University of Management and Technology, Lahore.

## Project Overview

TayyabBot uses a personal/custom dataset containing information about:

- Education
- Skills
- Projects
- Courses
- Certifications
- Professional experience
- Personal profile
- Resume

The chatbot retrieves relevant information from this dataset and uses a Large Language Model to generate contextual responses.

## RAG Pipeline

The system follows this pipeline:

Personal Dataset
       ↓
Preprocessing
       ↓
Chunking
       ↓
FastEmbed
       ↓
FAISS
       ↓
Retrieval
       ↓
Groq LLM
       ↓
Prompt + Conversation History
       ↓
Streamlit

## Technologies Used

- Python
- OpenAI API
- FAISS
- NumPy
- Streamlit
- python-dotenv
- uv

## Project Structure

```text
Tayyab_Chatbot/
│
├── dataset/
│   ├── personal_profile.txt
│   ├── education.txt
│   ├── skills.txt
│   ├── projects.txt
│   ├── courses.txt
│   ├── certifications.txt
│   ├── experience.txt
│   └── resume.txt
│
├── vectorstore/
│   ├── index.faiss
│   └── metadata.pkl
│
├── app.py
├── ingest.py
├── rag_pipeline.py
├── config.py
├── pyproject.toml
├── requirements.txt
├── .env
├── .gitignore
└── README.md
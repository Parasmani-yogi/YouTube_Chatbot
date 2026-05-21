# YouTube Video Q&A Chatbot (RAG)

A Streamlit-based Retrieval-Augmented Generation (RAG) chatbot for asking questions about YouTube videos using video transcripts.  
It fetches captions, splits them into chunks, indexes with FAISS, and answers user questions via an LLM in transcript context.

### Built using:

- Streamlit
- LangChain
- OpenAI
- FAISS
- YouTube Transcript API

---

# Features

- Extract transcript from YouTube videos
- Supports YouTube URLs and Video IDs
- Multi-language transcript support
- RAG-based question answering
- FAISS vector database
- MMR retrieval for better context diversity
- Chat interface with conversation history
- Transcript viewer
- Chunk information viewer
- Streamlit deployment ready

---

# Tech Stack

- Python
- Streamlit
- LangChain
- OpenAI Embeddings
- FAISS Vector Store
- YouTube Transcript API

---

# Installation

## 1. Clone Repository

```bash
git clone https://github.com/Parasmani-yogi/YouTube_Chatbot.git
cd YouTube_Chatbot
```

---

## 2. Create Virtual Environment Using UV

```bash
uv venv
```

Activate environment:

### Windows

```bash
.venv\Scripts\activate
```

### Linux/Mac

```bash
source .venv/bin/activate
```

---

## 3. Install Dependencies

Using UV:

```bash
uv pip install -r requirements.txt
```

OR

```bash
uv add langchain-openai langchain-community youtube-transcript-api faiss-cpu tiktoken python-dotenv openai streamlit
```

---

# Environment Variables

Create a `.env` file in the root directory.

```env
OPENAI_API_KEY=your_openai_api_key
```

---

# Run Streamlit App

```bash
streamlit run app.py
```

---

# How It Works

1. User enters a YouTube URL or Video ID
2. Transcript is extracted using YouTube Transcript API
3. Transcript is split into chunks
4. OpenAI embeddings are created
5. FAISS vector database stores embeddings
6. MMR retriever retrieves relevant chunks
7. GPT model answers user questions using retrieved context

---

# Retrieval Strategy

This project uses:

```python
search_type="mmr"
```

with:

```python
k=6
fetch_k=30
```

to improve retrieval diversity and reduce repetitive chunks.

---


# Deployment

This project can be deployed easily on:

- Streamlit Community Cloud
---

# Future Improvements

- Chat memory
- Multi-video support
- PDF export
- Whisper fallback for videos without subtitles
- Hybrid search
- Reranking
- Conversation summarization

---

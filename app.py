import re
from urllib.parse import urlparse, parse_qs

import streamlit as st
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


def extract_video_id(value: str) -> str | None:
    value = value.strip()

    # Raw 11-char YouTube ID
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", value):
        return value

    try:
        parsed = urlparse(value)

        # youtu.be/<id>
        if "youtu.be" in parsed.netloc:
            candidate = parsed.path.strip("/").split("/")[0]
            if re.fullmatch(r"[A-Za-z0-9_-]{11}", candidate):
                return candidate

        # youtube.com/watch?v=<id>
        if "youtube.com" in parsed.netloc or "m.youtube.com" in parsed.netloc:
            q = parse_qs(parsed.query).get("v", [None])[0]
            if q and re.fullmatch(r"[A-Za-z0-9_-]{11}", q):
                return q

            # youtube.com/shorts/<id> or /embed/<id>
            parts = [p for p in parsed.path.split("/") if p]
            if len(parts) >= 2 and parts[0] in {"shorts", "embed"}:
                candidate = parts[1]
                if re.fullmatch(r"[A-Za-z0-9_-]{11}", candidate):
                    return candidate
    except Exception:
        return None

    return None


def fetch_transcript_for_video(video_id: str):
    ytt_api = YouTubeTranscriptApi()

    # Prefer English, then Hindi, then any available transcript.
    for lang_code, lang_name in [("en", "English"), ("hi", "Hindi")]:
        try:
            data = ytt_api.fetch(video_id, languages=[lang_code])
            text = " ".join(chunk.text for chunk in data)
            if text.strip():
                return text, lang_name
        except Exception:
            pass

    try:
        data = ytt_api.fetch(video_id)
        text = " ".join(chunk.text for chunk in data)
        if text.strip():
            return text, "Auto"
    except Exception:
        pass

    raise TranscriptsDisabled("No transcripts available for this video.")


def normalize_question_for_retrieval(q: str) -> str:
    q = q.strip()
    if len(q.split()) <= 4:
        return f"{q}. Explain the concept with key points."
    return q


def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)


st.set_page_config(page_title="YouTube Chatbot - RAG", layout="wide")
st.title("YouTube Video Q&A Chatbot")
st.markdown("---")

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "main_chain" not in st.session_state:
    st.session_state.main_chain = None
if "video_id" not in st.session_state:
    st.session_state.video_id = None
if "transcript" not in st.session_state:
    st.session_state.transcript = None
if "transcript_language" not in st.session_state:
    st.session_state.transcript_language = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.sidebar.header("Video Setup")
video_input = st.sidebar.text_input(
    "YouTube Input",
    placeholder="Enter YouTube URL or Video ID",
    help="Paste the full YouTube URL or just the video ID",
    label_visibility="collapsed",
)

if st.sidebar.button("Load Video & Build RAG", key="load_video"):
    if not video_input.strip():
        st.error("Please enter a YouTube URL or Video ID")
        st.stop()

    video_id = extract_video_id(video_input)
    if not video_id:
        st.error("Invalid YouTube URL or Video ID")
        st.stop()

    with st.spinner("Fetching transcript..."):
        try:
            transcript, language_used = fetch_transcript_for_video(video_id)
        except TranscriptsDisabled:
            st.error("No captions available for this video.")
            st.stop()
        except Exception as e:
            st.error(f"Error fetching transcript: {e}")
            st.stop()

    with st.spinner("Preparing knowledge base..."):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.create_documents([transcript])

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vector_store = FAISS.from_documents(chunks, embeddings)

        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 6, "fetch_k": 30},
        )

        if language_used == "English":
            prompt_template = """
You are a transcript-grounded assistant for YouTube video Q&A.

Rules:
1. Use only the transcript context below.
2. Do not use outside knowledge.
3. If context is insufficient, reply exactly:
    I don't know based on the provided transcript.
4. Keep answers concise and clear.
5. Reply in the same language as the user's question.

Transcript Context:
{context}

Question:
{question}

Answer:
"""
        else:
            prompt_template = f"""
You are a transcript-grounded assistant for YouTube video Q&A.

Transcript primary language: {language_used}

Rules:
1. Use only the transcript context below.
2. Do not use outside knowledge.
3. If context is insufficient, reply exactly:
    I don't know based on the provided transcript.
4. Keep answers concise and clear.
5. For every response, provide exactly these two sections:
    - {language_used} Answer:
    - English Answer:
6. Both sections must contain the same meaning.

Transcript Context:
{{context}}

Question:
{{question}}

Answer:
"""

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"],
        )

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

        parallel_chain = RunnableParallel(
            {
                "context": RunnableLambda(normalize_question_for_retrieval)
                | retriever
                | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
        )

        parser = StrOutputParser()
        main_chain = parallel_chain | prompt | llm | parser

    st.session_state.vector_store = vector_store
    st.session_state.main_chain = main_chain
    st.session_state.transcript = transcript
    st.session_state.transcript_language = language_used
    st.session_state.video_id = video_id
    st.session_state.chat_history = []

    st.success("Video loaded. Start chatting below.")

if st.session_state.video_id:
    st.info(
        f"Loaded video ID: {st.session_state.video_id} | Transcript language: {st.session_state.transcript_language}"
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Show Transcript"):
            with st.expander("Full Transcript", expanded=False):
                st.text_area(
                    "Transcript Content",
                    value=st.session_state.transcript,
                    height=300,
                    disabled=True,
                )

    with col2:
        if st.button("Show Chunks Info"):
            if st.session_state.vector_store:
                with st.expander("Chunking Information", expanded=False):
                    st.write(f"Total Chunks: {len(st.session_state.vector_store.index_to_docstore_id)}")
                    st.write("Chunk Size: 1000")
                    st.write("Chunk Overlap: 200")
                    st.write("Search Type: MMR")
                    st.write("Top K Results: 6")

    st.markdown("---")
    st.header("Chat with Video")

    for message in st.session_state.chat_history:
        st.chat_message(message["role"]).write(message["content"])

    question = st.chat_input("Ask a question about the video...")

    if question:
        if st.session_state.main_chain is None:
            st.error("Please load a video first")
            st.stop()

        st.session_state.chat_history.append({"role": "user", "content": question})
        st.chat_message("user").write(question)

        with st.spinner("Thinking..."):
            try:
                answer = st.session_state.main_chain.invoke(question)
            except Exception as e:
                answer = f"Error generating answer: {e}"

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)

else:
    st.info("Please enter a YouTube URL or Video ID in the sidebar to get started.")
    st.markdown("---")
    st.write("Supported transcript languages include English, Hindi, and other available captions.")
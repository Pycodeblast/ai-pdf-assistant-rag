import streamlit as st
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline


# ---------------- UI CONFIG ---------------- #

st.set_page_config(
    page_title="AI PDF Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI PDF Assistant")
st.markdown("Ask questions from PDFs using **RAG + LangChain + LLM**")

# ---------------- SIDEBAR ---------------- #

with st.sidebar:

    st.header("⚙ Settings")

    chunk_size = st.slider(
        "Chunk Size",
        200,
        1000,
        500
    )

    chunk_overlap = st.slider(
        "Chunk Overlap",
        0,
        200,
        50
    )

    st.markdown("---")

    st.subheader("Tech Stack")

    st.write("LLM: DistilGPT2")
    st.write("Embeddings: MiniLM")
    st.write("Vector DB: FAISS")
    st.write("Framework: LangChain")


# ---------------- FILE UPLOAD ---------------- #

uploaded_files = st.file_uploader(
    "Upload PDF files",
    type="pdf",
    accept_multiple_files=True
)

question = st.text_input(
    "Ask a question about the PDFs"
)


# ---------------- PROCESS DOCUMENTS ---------------- #

if uploaded_files:

    docs = []

    for file in uploaded_files:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp:

            tmp.write(file.read())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)

        docs.extend(loader.load())

    splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    split_docs = splitter.split_documents(docs)


# ---------------- EMBEDDINGS ---------------- #

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )


# ---------------- VECTOR DATABASE ---------------- #

    vectorstore = FAISS.from_documents(
        split_docs,
        embeddings
    )

    retriever = vectorstore.as_retriever()


# ---------------- LLM ---------------- #

    pipe = pipeline(
    "text-generation",
    model="distilgpt2",
    max_new_tokens=150,
    truncation=True
)

    llm = HuggingFacePipeline(pipeline=pipe)


# ---------------- QUESTION ANSWERING ---------------- #

    if question:

        with st.spinner("Thinking..."):

            relevant_docs = retriever.invoke(question)

            context = "\n".join(
                [doc.page_content for doc in relevant_docs]
            )

            prompt = f"""
Use the following context to answer the question.

Context:
{context}

Question:
{question}

Answer:
"""

            response = llm.invoke(prompt)

        st.subheader("📌 Answer")

        st.write(response)


# ---------------- SHOW RETRIEVED CONTEXT ---------------- #

        with st.expander("🔎 Retrieved Context"):

            for doc in relevant_docs:

                st.write(doc.page_content[:500])
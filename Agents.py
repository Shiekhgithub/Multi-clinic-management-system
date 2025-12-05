import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph, END
from typing_extensions import List, TypedDict
from langchain_huggingface import HuggingFaceEmbeddings  # ✅ NEW UPDATED IMPORT
import streamlit as st
from langchain_community.vectorstores import FAISS
from pathlib import Path


# ---------------------- ENV + LLM ----------------------
load_dotenv()

openrouter_key = os.getenv("OPENROUTER_API_KEY")

llm = ChatOpenAI(
    model_name="openai/gpt-oss-20b:free",
    api_key=openrouter_key,
    base_url="https://openrouter.ai/api/v1"
)

# Updated embeddings (removes deprecation warning)
emb = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    encode_kwargs={"normalize_embeddings": True}
)

faiss_index = None


# ---------------------- File Type Detector ----------------------
def detect_file_type(file_path: str):
    return Path(file_path).suffix.lower()


# ---------------------- Loaders ----------------------
def Pdf_loader(filepath: str):
    pdf = PyPDFLoader(filepath)
    pdf_docs = pdf.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    parts = splitter.split_documents(pdf_docs)
    pdf_index = FAISS.from_documents(parts, emb)
    pdf_index.save_local("pdf-index")


def Text_loader(filepath: str):
    txt = TextLoader(filepath)
    txt_docs = txt.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    parts = splitter.split_documents(txt_docs)
    txt_index = FAISS.from_documents(parts, emb)
    txt_index.save_local("text-index")


def Md_loader(filepath: str):
    md = UnstructuredMarkdownLoader(filepath)
    md_docs = md.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    parts = splitter.split_documents(md_docs)
    md_index = FAISS.from_documents(parts, emb)
    md_index.save_local("md-index")


# ---------------------- STATE ----------------------
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# ---------------------- Document Loader ----------------------
def Load_Docs(filename: str):
    global faiss_index

    file_type = detect_file_type(filename)

    if file_type == ".pdf":
        if not os.path.exists("pdf-index"):
            Pdf_loader(filename)
        faiss_index = FAISS.load_local("pdf-index", emb, allow_dangerous_deserialization=True)

    elif file_type == ".txt":
        if not os.path.exists("text-index"):
            Text_loader(filename)
        faiss_index = FAISS.load_local("text-index", emb, allow_dangerous_deserialization=True)

    elif file_type == ".md":
        if not os.path.exists("md-index"):
            Md_loader(filename)
        faiss_index = FAISS.load_local("md-index", emb, allow_dangerous_deserialization=True)

    else:
        raise ValueError("Unsupported file format")

    return faiss_index


# ---------------------- RAG FUNCTIONS ----------------------
def retrieve(state: State):
    results = faiss_index.similarity_search(state["question"])
    return {"context": results}


def generate(state: State):
    docs_text = "\n\n".join(doc.page_content for doc in state["context"])
    question = state["question"]

    prompt = f"""
You are an intelligent cardiology research assistant.

Use ONLY the provided context to answer.

Rules:
- DO NOT answer anything unrelated to heart diseases.
- If question is unrelated say: "I can just answer about Heart Related queries."
- Respond in clear and professional paragraphs.
- At the end ALWAYS add this line exactly:

Would you like to book appointment with our Cardiologist Dr Ahmed? Please click on the link below to book your appointment.

Context:
{docs_text}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)
    return {"answer": response.content}


# ---------------------- GRAPH ----------------------
Graph_builder = StateGraph(State)

Graph_builder.add_node("Retriever", retrieve)
Graph_builder.add_node("Generator", generate)

Graph_builder.add_edge(START, "Retriever")
Graph_builder.add_edge("Retriever", "Generator")
Graph_builder.add_edge("Generator", END)

graph = Graph_builder.compile()


# ---------------------- STREAMLIT UI ----------------------
st.title("🫀 AI Heart Disease Assistant")

uploaded_file = st.file_uploader("Upload PDF, TXT, or MD File", type=["pdf", "txt", "md"])

if uploaded_file:
    file_path = uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success("File uploaded successfully!")
    Load_Docs(file_path)

user_question = st.text_input("Ask a question related to Heart Disease:")

if st.button("Ask"):
    if not user_question.strip():
        st.warning("Please enter a question.")
    else:
        result = graph.invoke({"question": user_question})
        st.write(result["answer"])
        st.write("\n")  # ✅ NEW LINE FIX FOR STREAMLIT

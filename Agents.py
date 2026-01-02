import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph, END
from typing_extensions import List, TypedDict
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from pathlib import Path


# ---------------------- ENV + LLM ----------------------
load_dotenv()

openrouter_key = os.getenv("OPENROUTER_API_KEY")

# Using Meta Llama 3.2 - reliable free tier model
llm = ChatOpenAI(
    model_name="meta-llama/llama-3.2-3b-instruct:free",
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
    pdf_index.save_local("faiss-index")


def Text_loader(filepath: str):
    txt = TextLoader(filepath)
    txt_docs = txt.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    parts = splitter.split_documents(txt_docs)
    txt_index = FAISS.from_documents(parts, emb)
    txt_index.save_local("faiss-index")


def Md_loader(filepath: str):
    md = UnstructuredMarkdownLoader(filepath)
    md_docs = md.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    parts = splitter.split_documents(md_docs)
    md_index = FAISS.from_documents(parts, emb)
    md_index.save_local("faiss-index")


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
        Pdf_loader(filename)
        faiss_index = FAISS.load_local("faiss-index", emb, allow_dangerous_deserialization=True)

    elif file_type == ".txt":
        Text_loader(filename)
        faiss_index = FAISS.load_local("faiss-index", emb, allow_dangerous_deserialization=True)

    elif file_type == ".md":
        Md_loader(filename)
        faiss_index = FAISS.load_local("faiss-index", emb, allow_dangerous_deserialization=True)

    else:
        raise ValueError("Unsupported file format")

    return faiss_index


# ---------------------- RAG FUNCTIONS ----------------------
def retrieve(state: State):
    global faiss_index
    # Load the index if not already loaded
    if faiss_index is None and os.path.exists("faiss-index"):
        faiss_index = FAISS.load_local("faiss-index", emb, allow_dangerous_deserialization=True)
    
    if faiss_index is None:
        raise ValueError("No document has been uploaded yet. Please upload a document first.")
    
    results = faiss_index.similarity_search(state["question"])
    return {"context": results}


def generate(state: State):
    docs_text = "\n\n".join(doc.page_content for doc in state["context"])
    question = state["question"]

    prompt = f"""
You are an intelligent medical assistant for a multi-clinic management system. You help clinic staff (doctors, receptionists, administrators) and patients by providing accurate information based on uploaded medical documents and clinic resources.

IMPORTANT RULES:
1. Answer ONLY based on the provided context below
2. If the question is not related to the uploaded medical documents or clinical information, politely decline and say: "I can only answer questions based on the uploaded medical documents and clinical information in our system."
3. Provide clear, professional, and accurate responses
4. If you cannot find the answer in the context, say: "I don't have that specific information in the uploaded documents. Please consult with your healthcare provider or clinic staff for more details."
5. Never make up medical information or provide advice beyond what's in the documents
6. Be helpful to all user types: patients seeking information, clinic staff looking up protocols, or administrators reviewing resources

Context from uploaded documents:
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

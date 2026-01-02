# Multi-Clinic Management System - RAG Chat Agent

A Retrieval-Augmented Generation (RAG) chat application that allows users to upload documents (PDF, TXT, or MD) and ask questions about their content using AI.

## Features

- Upload PDF, TXT, or Markdown files
- Ask questions about uploaded documents
- AI-powered responses using OpenRouter API
- Built with FastAPI backend and Streamlit frontend

## Prerequisites

- Python 3.10 or higher
- OpenRouter API key ([Get one here](https://openrouter.ai/))

## Installation

1. **Clone or navigate to the project directory**
   ```bash
   cd Multi-clinic-management-system
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   
   Create a `.env` file in the project root directory:
   ```
   OPENROUTER_API_KEY=your_openrouter_api_key_here
   ```

## Running the Application

The application consists of two parts that need to run simultaneously:

### 1. Start the FastAPI Backend

Open a terminal and run:
```bash
uvicorn Backend:app --reload
```

The backend will start on `http://127.0.0.1:8000`

### 2. Start the Streamlit Frontend

Open a **second terminal** and run:
```bash
streamlit run Streamlit.py
```

The Streamlit app will automatically open in your browser at `http://localhost:8501`

## Usage

1. **Upload a Document**: Use the file uploader to upload a PDF, TXT, or MD file
2. **Ask Questions**: Type your questions about the document in the chat input
3. **Get AI Responses**: The system will retrieve relevant information and provide answers

## Project Structure

- `Agents.py` - LangChain agents and document processing logic
- `Backend.py` - FastAPI server with file upload and chat endpoints
- `Streamlit.py` - Streamlit frontend interface
- `requirements.txt` - Python dependencies
- `.env` - Environment variables (create this file)

## Troubleshooting

- **ModuleNotFoundError**: Make sure all dependencies are installed with `pip install -r requirements.txt`
- **API Key Error**: Verify your `.env` file contains a valid `OPENROUTER_API_KEY`
- **Connection Error**: Ensure the FastAPI backend is running before using the Streamlit frontend

## Technologies Used

- **FastAPI** - Backend API framework
- **Streamlit** - Frontend UI
- **LangChain** - LLM orchestration
- **FAISS** - Vector database for document embeddings
- **HuggingFace** - Embedding models
- **OpenRouter** - LLM API

from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from Agents import graph, Load_Docs
import tempfile
from typing import List

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)


class user_entry(BaseModel):
    question: str


@app.post("/Upload_File")
async def upload(files: List[UploadFile]):
    processed_files = []
    temp_paths = []
    
    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as temp:
            temp.write(await file.read())
            temp_path = temp.name
            temp_paths.append(temp_path)
            processed_files.append(file.filename)
    
    # Load all documents into the same index
    Load_Docs(temp_paths)

    return {
        "filenames": processed_files,
        "count": len(processed_files),
        "status": "All files processed successfully",
    }


@app.post("/chat")
async def chat(user: user_entry):
    try:
        answer = graph.invoke({"question": user.question})
        result = answer["answer"]

        return {
            "Assistant": result
        }
    except Exception as e:
        error_msg = str(e)
        
        # Handle specific error types
        if "AuthenticationError" in str(type(e)) or "401" in error_msg:
            return {
                "Assistant": "⚠️ API Authentication Error: Please check your OpenRouter API key or add credits to your account. Visit https://openrouter.ai/ to manage your account."
            }
        elif "RateLimitError" in str(type(e)) or "429" in error_msg:
            return {
                "Assistant": "⚠️ Rate limit exceeded. Please wait a moment and try again."
            }
        elif "No document has been uploaded" in error_msg:
            return {
                "Assistant": "⚠️ Please upload a document first before asking questions."
            }
        else:
            return {
                "Assistant": f"⚠️ An error occurred: {error_msg}. Please try again or contact support if the issue persists."
            }


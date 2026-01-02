from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from Agents import graph, Load_Docs
import tempfile

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
async def upload(file: UploadFile):
    with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as temp:
        temp.write(await file.read())
        temp_path = temp.name

    Load_Docs(temp_path)

    return {
        "filename": file.filename,
        "stored_path": temp_path,
        "status": "Processed successfully",
    }


@app.post("/chat")
async def chat(user: user_entry):
    answer = graph.invoke({"question": user.question})
    result = answer["answer"]

    return {
        "Assistant": result
    }

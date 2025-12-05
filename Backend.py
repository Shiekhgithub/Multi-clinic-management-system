from fastapi import FastAPI, Request, Depends, HTTPException, UploadFile, File
from pydantic import BaseModel
from Agents import graph, Load_Docs
import tempfile

app = FastAPI()

class user_entry(BaseModel):
    question : str


@app.post("/Upload_File")
async def upload(file : UploadFile):
    with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as temp:
        temp.write(await file.read())
        temp_path = temp.name

    faiss_index = Load_Docs(temp_path)

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

from fastapi import FastAPI
# Pydantic is a data validation library
# BaseModel is used to define request/response structure
# Same concept as a DTO in Spring Boot
from pydantic import BaseModel
from rag import load_qa_chain #importing function from rag.py
from contextlib2 import asynccontextmanager

# UploadFile — FastAPI's class for handling file uploads
# File — tells FastAPI to expect a file in the request
# shutil — Python built-in for file operations like copying
# os — to create folders and handle paths
from fastapi import FastAPI, UploadFile, File
import shutil
import os

@asynccontextmanager
async def lifespan(app: FastAPI):
    global chain
    chain = load_qa_chain()
    print("RAG chain loaded and ready!")
    yield

app = FastAPI(title="RAG Chatbot", lifespan=lifespan)
chain = None

# BaseModel automatically validates the request — if text is missing it returns an error
# str means text must be a string
class Question(BaseModel):
    text: str

@app.get("/")
def root():
    return {"status": "RAG Chatbot is running"}

@app.post("/ask")
def ask(q: Question):
    if chain is None:
        return {"error": "Chain not loaded yet. Try again in a moment."}
    answer = chain.invoke(q.text)
    return {"answer": answer.strip()}

# async def — this function is asynchronous because file upload involves waiting
# UploadFile = File(...) — expects a file in the request, ... means required
# os.makedirs("./docs", exist_ok=True) — creates docs/ folder if it doesn't exist, exist_ok=True means no error if it already exists
# f"./docs/{file.filename}" — builds the save path using the original filename
# open(file_path, "wb") — opens file for writing in binary mode — wb = write binary, same as any file in Java
# shutil.copyfileobj — copies the uploaded file content to disk
# Returns a message telling user to call /ingest next
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    os.makedirs("./docs", exist_ok=True)
    file_path = f"./docs/{file.filename}"
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"message": f"{file.filename} uploaded successfully. Now call /ingest."}

# @app.post("/ingest") — handles POST requests to /ingest
# global chain — we're updating the global chain variable
# ingest() — runs the full ingest pipeline on all PDFs in docs/ folder
# chain = load_qa_chain() — reloads the chain with the newly indexed PDFs
# Returns confirmation message
@app.post("/ingest")
def run_ingest():
    global chain
    ingest()
    chain = load_qa_chain()
    return {"message": "Ingestion complete. You can now ask questions."}

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import shutil
from utils import *
from uuid import uuid4
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi import FastAPI, Response, Request, File, UploadFile

import chromadb
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents.stuff import StuffDocumentsChain


app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

directory = 'index_store'
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
llm = Ollama(model="llama2", base_url="http://127.0.0.1:11434", verbose=True, temperature=0.6, )
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
client = chromadb.PersistentClient(path=directory)


prompt_template = """Use the following pieces of context to answer the question at the end. Please follow the following rules:
1. If the question is to request links, please only return the source links with no answer.
2. If you don't know the answer, don't try to make up an answer. Just say **I can't find the final answer but you may want to check the following links** and add the source links as a list.
3. If you find the answer, write the answer in a concise way and add the list of sources that are **directly** used to derive the answer. Exclude the sources that are irrelevant to the final answer.

{context}

Question: {question}
Helpful Answer:"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt_template)
llm_chain = LLMChain(llm=llm, prompt=QA_CHAIN_PROMPT, callbacks=None, verbose=True)
document_prompt = PromptTemplate(
        input_variables=["page_content", "source"],
        template="Context:\ncontent:{page_content}\nsource:{source}",
    )

combine_documents_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context",
        document_prompt=document_prompt,
        callbacks=None,
    )


@app.get("/")
async def index(request: Request, response: Response):
    response = templates.TemplateResponse(request=request, name='index.html', response=response)
    response.set_cookie("cookie", uuid4())
    return response


@app.post("/upload/")
async def upload(request: Request, response: Response, file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        return JSONResponse({"error": "File not allowded!"})

    if 'cookie' not in request.cookies.keys():
        return JSONResponse({"error": "Cookie not found!"})

    session_cookie = request.cookies.get('cookie')
    
    file_location = f"./media/{file.filename}"
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)    

    loader = PyPDFLoader(file_location)
    pages = loader.load_and_split(text_splitter)
    
    vector_index = Chroma.from_documents(documents=pages, embedding=embeddings, persist_directory=directory, collection_name=create_hash(session_cookie))    
    vector_index.persist()

    redirect_url = request.url_for('chat')
    return JSONResponse({"filename": file.filename, "redirect_url": redirect_url.__str__()})


@app.get('/chat')
async def chat(request: Request, response: Response):
    return templates.TemplateResponse(request=request, name='chat.html', response=response)


@app.post('/message')
async def message(request: Request, response: Response):
    if 'cookie' not in request.cookies.keys():
        return JSONResponse({"error": "Cookie not found!"})

    session_cookie = request.cookies.get('cookie')
    vectordb = Chroma(persist_directory=directory, embedding_function=embeddings, collection_name=create_hash(session_cookie))
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k":6})

    data = await request.form()
    message = data.get('message')

    qa = RetrievalQA(
        combine_documents_chain=combine_documents_chain,
        callbacks=None,
        verbose=False,
        retriever=retriever,
        return_source_documents=False,
    )    
    response = qa.invoke(message)
    return JSONResponse(response)

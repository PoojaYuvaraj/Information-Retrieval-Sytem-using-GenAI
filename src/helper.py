import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import GooglePalmEmbeddings
# from langchain.llms import GooglePalm
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chat_models import ChatHuggingFace
# from huggingface_hub import InferenceClient
from langchain.llms import HuggingFaceHub
from huggingface_hub import login
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline, AutoModelForCausalLM
from openai import ChatCompletion
from langchain.chat_models import ChatOpenAI

import torch


load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")
huggingface_gemma = os.getenv("HUGGINGFACE_GEMMA")
openai_api_key = os.getenv("OPENAI_API_KEY")


os.environ["GOOGLE_API_KEY"] = google_api_key
os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_api_key
os.environ["HUGGINGFACE_GEMMA"] = huggingface_gemma
os.environ["OPENAI_API_KEY"] = openai_api_key
login(huggingface_gemma)

os.environ["STREAMLIT_WATCH_USE_POLLING"] = "true"
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text 

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    # embeddings = GooglePalmEmbeddings(google_api_key=google_api_key)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks,embedding = embeddings)
    return vector_store

def get_conversational_chain(vector_store):
    # Use Langchain's wrapper for OpenAI Chat models
    llm = ChatOpenAI(model="gpt-4.1-mini", openai_api_key=openai_api_key)
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conversation_chain


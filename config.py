import os
from dotenv import load_dotenv

load_dotenv()

# Embedding Model and Vector Database Settings
EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
VECTOR_STORE_DIR = "./vectorstore/"
COLLECTION_NAME = "interview_genie_qa"

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_NAME = "mixtral-8x7b-32768"

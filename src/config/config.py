import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    LLM_MODEL = "llama-3.1-8b-instant"

    EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 100

    @classmethod
    def get_llm(cls):
        if not cls.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not set in .env")

        return init_chat_model(
            model=cls.LLM_MODEL,
            model_provider="groq",
            api_key=cls.GROQ_API_KEY,
            temperature=0
        )
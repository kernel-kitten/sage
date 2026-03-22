import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

class VectorStore:
    def __init__(self, model_name):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        self.vectorstore = None

    def create_or_load(self, documents, index_path):
        if os.path.exists(index_path):
            self.vectorstore = FAISS.load_local(
                index_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            self.vectorstore = FAISS.from_documents(documents, self.embeddings)
            self.vectorstore.save_local(index_path)

    def get_retriever(self):
        return self.vectorstore.as_retriever(search_kwargs={"k": 4})
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path

class DocumentProcessor:
    def __init__(self, chunk_size=500, chunk_overlap=100):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def load_documents(self, folder_path):
        docs = []
        folder = Path(folder_path)

        for file in folder.glob("*.txt"):
            loader = TextLoader(str(file), encoding="utf-8")
            docs.extend(loader.load())

        return docs

    def split_documents(self, documents):
        return self.splitter.split_documents(documents)

    def process(self, folder_path):
        docs = self.load_documents(folder_path)
        return self.split_documents(docs)
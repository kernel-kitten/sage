from src.config.config import Config
from src.document_processor.document_processor import DocumentProcessor
from src.vectorstore.vectorstore import VectorStore

class SimpleRAG:
    def __init__(self, subject_name, subject_path):
        self.subject_name = subject_name

        self.llm = Config.get_llm()
        self.processor = DocumentProcessor(
            Config.CHUNK_SIZE,
            Config.CHUNK_OVERLAP
        )

        self.vectorstore = VectorStore(Config.EMBEDDING_MODEL)

        # Process docs
        documents = self.processor.process(subject_path)

        # Unique index per subject
        index_path = f"faiss_{subject_name}"

        self.vectorstore.create_or_load(documents, index_path)

        self.retriever = self.vectorstore.get_retriever()

    def ask(self, question: str):
        docs = self.retriever.invoke(question)

        context = "\n\n".join([d.page_content for d in docs])

        prompt = f"""
You are a helpful study assistant.

Answer ONLY from the context below.
If answer is not present, say "Not found in notes".

Context:
{context}

Question:
{question}
"""

        response = self.llm.invoke(prompt)

        return response.content, docs
import shutil
import os
from typing import Any

from dotenv import load_dotenv

from langchain_community.document_loaders import YoutubeLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama

load_dotenv()

class SimpleRAG:
    def __init__(self):
        self.embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.llm_model_name = os.getenv("OLLAMA_MODEL", "llama3.2")
        self.vector_store: Chroma | None = None
        self.generator: Any | None = None

    # Load Youtube Transcript
    def get_transcript(self, url):
        try:
            loader = YoutubeLoader.from_youtube_url(
                url,
                add_video_info=False
            )
            return loader.load()
        except Exception as e:
            print("Error loading transcript:", e)
            return []
    # Split document(Transcripts)
    def split_document(self, documents):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
        )
        return splitter.split_documents(documents)
    # Create Vector Store
    def create_vector_store(self, chunks):
        if not chunks:
            raise ValueError("No chunks to create vector store.")
        embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
        )
        if os.path.exists("./chroma_db"):
            shutil.rmtree("./chroma_db")
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="./chroma_db",
        )

    # Load LLM running through Ollama
    def load_llm(self):
        return ChatOllama(
            model=self.llm_model_name,
            base_url="http://localhost:11434",
            temperature=0.7
        )

    # Setup Pipeline
    def setup(self, url):
        transcript_docs = self.get_transcript(url)
        print("Transcript docs:", len(transcript_docs))

        chunks = self.split_document(transcript_docs)
        print("Chunks:", len(chunks))

        self.create_vector_store(chunks)
        print("Vector store created:", self.vector_store is not None)

        self.generator = self.load_llm()
        print("LLM loaded:", self.generator is not None)

    # Ask Questions
    def ask(self, question: str, k: int = 3):
        if self.vector_store is None or self.generator is None:
            raise ValueError("Run setup() before asking questions.")

        retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
        docs = retriever.invoke(question)

        if not docs:
            return "No relevant information found in the video."
        
        context = "\n\n".join([
            f"[Source: {doc.metadata.get('title', 'video')}]\n{doc.page_content}"
            for doc in docs
        ])

        prompt = f"""You are an AI assistant helping a student understand a YouTube lecture.
        Use ONLY the provided context to answer.
        If the answer is not in the context, say "I don't know based on the video".
        Explain clearly and simply.
        Context:
        {context}
        Question:
        {question}
        Answer:
        """

        response = self.generator.invoke(prompt)
        return response.content

if __name__ == "__main__":
    rag = SimpleRAG()
    
    url = input("Enter YouTube URL: ")
    rag.setup(url)

    while True:
        q = input("Ask: ")
        if q.lower() == "exit":
            break
        print(rag.ask(q))

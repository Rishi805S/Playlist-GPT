import shutil
import os
from typing import Any
import hashlib
import numpy as np

from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi
# from langchain_community.document_loaders import YoutubeLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
# from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_core.documents import Document

load_dotenv()

class SimpleRAG:
    def __init__(self):
        # self.embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.embedding_model_name = "sentence-transformers/all-MiniLM-L12-v2" # Slighly faster
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name
        )
        self.llm_model_name = os.getenv("OLLAMA_MODEL", "llama3.2")
        self.vector_store: Chroma | None = None
        self.generator: Any | None = None
        self.video_id = None

    # Load Youtube Transcript
    # Currently this Youtube loader is only giving content not timestamps so we use youtube-transcript-api
    # def get_transcript(self, url):
    #     try:
    #         loader = YoutubeLoader.from_youtube_url(
    #             url,
    #             add_video_info=False
    #         )
    #         return loader.load()
    #     except Exception as e:
    #         print("Error loading transcript:", e)
    #         return []


    def get_transcript(self, url):
        try:
            video_id = url.split("v=")[-1]
            # This is an older way of fetching youtube transcript
            # transcript = YouTubeTranscriptApi.get_transcript(video_id)
            api = YouTubeTranscriptApi()
            data = api.fetch(video_id)

            raw_data = data.to_raw_data() # Gives transcript in raw json format having content, start, duration 
            documents = []

            for item in raw_data:
                doc = {
                    "text": item["text"],
                    "start": item["start"],
                    "end": item["start"] + item["duration"]
                }
                documents.append(doc)

            return documents

        except Exception as e:
            print("Error loading transcript:", e)
            return []
        
    def chunk_transcript(self, docs, chunk_size=10, overlap=3):
        chunks = []
        
        for i in range(0, len(docs), chunk_size - overlap):
            chunk_docs = docs[i:i + chunk_size]
            
            if not chunk_docs:
                continue
            
            chunks.append({
                "text": " ".join([d["text"] for d in chunk_docs]),
                "start": chunk_docs[0]["start"],
                "end": chunk_docs[-1]["end"]
            })
        
        return chunks
        
    # Split document(Transcripts)----> Currently not using this becuase I need to chunk with respect to timetamps this current splitter will chunk based on words 
    # def split_document(self, documents):
    #     splitter = RecursiveCharacterTextSplitter(
    #         chunk_size=700,
    #         chunk_overlap=50,
    #     )
    #     return splitter.split_documents(documents)

    # Instead of using RecuriveCharacterTextSplitter, I used this custom function to convert to Document
    def convert_to_documents(self, chunks):
        documents = []

        for chunk in chunks:
            doc = Document(
                page_content=chunk["text"],
                metadata={
                    "start": chunk["start"],
                    "end": chunk["end"]
                }
            )
            documents.append(doc)

        return documents
    
    # This is used to get the path of video in chrome_db
    def get_db_path(self):
        return f"./chroma_db_{self.video_id}"
    
    # Create Vector Store
    def create_vector_store(self, chunks):
        embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
        )

        db_path = self.get_db_path()

        if os.path.exists(db_path):
            print("Loading existing vector DB...")
            self.vector_store = Chroma(
                persist_directory=db_path,
                embedding_function=embeddings,
            )
            # self.vector_store.persist() used this before but now chroma_db auto saves so we don't need this
        else:
            print("Creating new vector DB...")
            self.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=db_path,
            )

    # Load LLM running through Ollama before used Huggingface Inference Endpoints but having issues like connectivity error so switched
    def load_llm(self):
        return ChatOllama(
            model=self.llm_model_name,
            base_url="http://localhost:11434",
            temperature=0.3
        )
    
    def get_candidate_lines(self, retrieved_docs):
        candidates = []

        for doc in retrieved_docs:
            chunk_text = doc.page_content.lower()

            for line in self.transcript_docs:
                if line["text"].lower() in chunk_text:
                    candidates.append(line)

        return candidates
    
    

    def find_best_timestamp(self, query, candidates):
        if not candidates:
            return None

        texts = [c["text"] for c in candidates]

        # Embed
        doc_embeddings = self.embedding_model.embed_documents(texts)
        query_embedding = self.embedding_model.embed_query(query)

        # Cosine similarity manually (no sklearn needed)
        similarities = []

        for emb in doc_embeddings:
            sim = np.dot(query_embedding, emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(emb)
            )
            similarities.append(sim)

        best_idx = int(np.argmax(similarities))

        return candidates[best_idx]["start"]

    # Setup Pipeline
    def setup(self, url):
        self.video_id = url.split("v=")[-1]
        
        transcript_docs = self.get_transcript(url)
        self.transcript_docs = transcript_docs
        if not transcript_docs:
            raise ValueError("Transcript could not be loaded.")
        print("Transcript docs:", len(transcript_docs))

        raw_chunks = self.chunk_transcript(transcript_docs)
        print("Chunks:", len(raw_chunks))

        documents = self.convert_to_documents(raw_chunks)
        

        self.create_vector_store(documents)
        print("Vector store created:", self.vector_store is not None)

        self.generator = self.load_llm()
        print("LLM loaded:", self.generator is not None)

    # Ask Questions
    def ask(self, question: str, k: int = 5):
        if self.vector_store is None or self.generator is None:
            raise ValueError("Run setup() before asking questions.")

        retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
        docs = retriever.invoke(question)
        candidates = self.get_candidate_lines(docs)
        best_time = self.find_best_timestamp(question, candidates)

        if not docs:
            return "No relevant information found in the video."
        
        # Currently this is not giving timestamps so we modify this
        # context = "\n\n".join([
        #     f"[Source: {doc.metadata.get('title', 'video')}]\n{doc.page_content}"
        #     for doc in docs
        # ])

        # This method fetches timestamps
        context = "\n\n".join([
            f"[Timestamp: {doc.metadata.get('start', 0)} seconds]\n{doc.page_content}"
            for doc in docs
        ])

        prompt = f"""You are an AI assistant helping a student understand a YouTube lecture.
        Use ONLY the provided context to answer.
        If the answer is not in the context, say "I don't know based on the video".
        When timestamps are present (in seconds), convert them into human readable format (like minutes:seconds).
        Explain clearly and simply.
        Context:
        {context}
        Question:
        {question}
        Answer:
        """

        response = self.generator.invoke(prompt)
        answer = response.content
        if best_time is not None:
            answer += f"\n\n(Approx timestamp: {int(best_time)} seconds)"
        return response.content

if __name__ == "__main__":
    rag = SimpleRAG()
    
    
    url = input("Enter YouTube URL: ")
    rag.setup(url)

    while True:
        q = input("ASK PLAYLISTGPT: ")
        if q.lower() == "exit":
            break
        print(rag.ask(q))

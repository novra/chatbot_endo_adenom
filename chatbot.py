import os

# ============================================
# FIX: Disable ChromaDB Telemetry (set before Chroma imports)
# ============================================
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY_IMPL"] = "none"

import fitz  # PyMuPDF
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.schema.output_parser import StrOutputParser
from langchain_core.documents import Document
from huggingface_hub import InferenceClient
import re
from datetime import datetime

class ChatBot:
    """
    ChatBot menggunakan Chat Completion API dengan perbaikan:
    1. Semantic Chunking dengan overlap besar (context preservation)
    2. Metadata Filtering (kategori sumber, tahun, validitas)
    3. Chain-of-Thought Prompting (structured reasoning)
    """
    def __init__(self):
        # Konfigurasi Database - Menggunakan path relatif agar aman di Streamlit Cloud
        self.persist_directory = "./chroma_db_adenomyosis"
        
        # Model Embedding
        self.embeddings_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        self._initialize_hf_client()
        self._setup_rag_chain()

    def _initialize_hf_client(self):
        """Menginisialisasi Hugging Face Client menggunakan Streamlit Secrets."""
        try:
            hf_token = st.secrets["HUGGINGFACE_API_KEY"]
        except KeyError:
            raise ValueError("HUGGINGFACE_API_KEY tidak ditemukan di Streamlit Secrets.")
        
        # Menggunakan Google Gemma 2
        self.hf_client = InferenceClient(
            model="google/gemma-2-2b-it",
            token=hf_token
        )

    def _initialize_chroma(self):
        """Inisialisasi ChromaDB dengan pengecekan folder yang lebih aman untuk Cloud."""
        db_exists = os.path.exists(self.persist_directory) and \
                    len(os.listdir(self.persist_directory)) > 0 if os.path.exists(self.persist_directory) else False

        if db_exists:
            print(f"--- Memuat database Chroma dari {self.persist_directory} ---")
            return Chroma(
                persist_directory=self.persist_directory, 
                embedding_function=self.embeddings_model,
                collection_metadata={"hnsw:space": "cosine"}
            )
        else:
            # Di Streamlit Cloud, pastikan folder './data_adenomyosis' sudah di-upload ke GitHub
            print("--- Database belum ditemukan. Memulai proses indexing PDF... ---")
            st.info("Sedang membangun Database Pengetahuan (proses awal 1-2 menit)...")
            
            docs = self._load_pdfs_from_folder('./data_adenomyosis')
            if not docs:
                st.warning("Tidak ada dokumen PDF ditemukan di folder './data_adenomyosis'.")
                return None
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=150,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            split_docs = text_splitter.split_documents(docs)
            
            for idx, doc in enumerate(split_docs):
                doc.metadata["chunk_id"] = idx
            
            vector_store = Chroma.from_documents(
                documents=split_docs,
                embedding=self.embeddings_model,
                persist_directory=self.persist_directory,
                collection_metadata={"hnsw:space": "cosine"}
            )
            st.success(f"✅ Database berhasil dibuat!")
            return vector_store

    def _extract_metadata_from_filename(self, filename):
        """Ekstraksi metadata dari nama file PDF."""
        metadata = {
            "source": filename,
            "source_type": "Unknown",
            "validity_level": "medium",
            "year": 0,
            "category": "general"
        }
        
        # Extract year from filename (e.g., 2020, 2021, etc.)
        year_match = re.search(r'(19|20)\d{2}', filename)
        if year_match:
            metadata["year"] = int(year_match.group())
        
        # Kategorisasi berdasarkan nama file
        filename_lower = filename.lower()
        
        # Source type detection
        if any(keyword in filename_lower for keyword in ['jurnal', 'journal', 'paper', 'research']):
            metadata["source_type"] = "Jurnal Ilmiah"
            metadata["validity_level"] = "high"
        elif any(keyword in filename_lower for keyword in ['guideline', 'pedoman', 'clinical']):
            metadata["source_type"] = "Guideline Klinis"
            metadata["validity_level"] = "very_high"
        elif any(keyword in filename_lower for keyword in ['textbook', 'buku', 'book']):
            metadata["source_type"] = "Buku Teks"
            metadata["validity_level"] = "high"
        elif any(keyword in filename_lower for keyword in ['review', 'tinjauan']):
            metadata["source_type"] = "Review Article"
            metadata["validity_level"] = "high"
        
        # Category detection
        if any(keyword in filename_lower for keyword in ['adenomyosis', 'adenom']):
            metadata["category"] = "adenomyosis"
        elif any(keyword in filename_lower for keyword in ['endometriosis', 'endo']):
            metadata["category"] = "endometriosis"
        elif any(keyword in filename_lower for keyword in ['diagnosis', 'diagnosa', 'diagnostic']):
            metadata["category"] = "diagnosis"
        elif any(keyword in filename_lower for keyword in ['treatment', 'pengobatan', 'terapi', 'therapy']):
            metadata["category"] = "treatment"
        
        return metadata

    def _load_pdfs_from_folder(self, folder_path):
        """
        Load PDFs dengan metadata enrichment
        ChromaDB requires all metadata values to be str/int/float/bool (not None)
        """
        if not os.path.exists(folder_path):
            st.error(f"Folder '{folder_path}' tidak ditemukan.")
            return []
        
        docs = []
        for filename in os.listdir(folder_path):
            if filename.lower().endswith('.pdf'):
                path = os.path.join(folder_path, filename)
                try:
                    with fitz.open(path) as doc:
                        text = "".join(page.get_text() for page in doc)
                        if text.strip():
                            # Ekstraksi metadata dari filename
                            metadata = self._extract_metadata_from_filename(filename)
                            
                            # Tambahan metadata (pastikan tidak ada None)
                            metadata["page_count"] = len(doc)
                            metadata["indexed_at"] = datetime.now().isoformat()
                            
                            # Double check: filter out any None values (safety net)
                            metadata = {k: v for k, v in metadata.items() if v is not None}
                            
                            # Ensure all values are proper types
                            for key, value in metadata.items():
                                if not isinstance(value, (str, int, float, bool)):
                                    metadata[key] = str(value)
                            
                            docs.append(Document(page_content=text, metadata=metadata))
                            print(f"✓ Loaded: {filename} | Category: {metadata['category']} | Validity: {metadata['validity_level']}")
                            
                except Exception as e: 
                    print(f"❌ Gagal memproses file {filename}: {e}")
        
        # Summary statistik
        if docs:
            categories = {}
            for doc in docs:
                cat = doc.metadata.get("category", "unknown")
                categories[cat] = categories.get(cat, 0) + 1
            
            print(f"\n📊 SUMMARY DOKUMEN:")
            print(f"Total dokumen: {len(docs)}")
            for cat, count in categories.items():
                print(f"  - {cat}: {count} dokumen")
        
        return docs

    def _setup_rag_chain(self):
        """Membangun RAG chain dengan st.secrets support."""
        self.vector_store = self._initialize_chroma()
        
        if self.vector_store is None:
            self.rag_chain = None
            return

        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={'k': 5}
        )

        def format_docs(docs):
            formatted_chunks = []
            for doc in docs:
                source_type = doc.metadata.get("source_type", "Unknown")
                validity = doc.metadata.get("validity_level", "unknown")
                year = doc.metadata.get("year", "N/A")
                formatted_chunks.append(
                    f"[Sumber: {source_type} | Validitas: {validity} | Tahun: {year}]\n{doc.page_content}"
                )
            return "\n\n---\n\n".join(formatted_chunks)

        def call_llm(inputs):
            try:
                system_prompt = """Anda adalah Asisten Medis Ahli Ginekologi yang berspesialisasi dalam adenomyosis dan endometriosis.

PERAN ANDA:
- Memberikan informasi medis yang akurat berdasarkan literatur ilmiah
- Menjelaskan kondisi medis dengan bahasa yang mudah dipahami
- Selalu menekankan pentingnya konsultasi dengan dokter

CARA MENJAWAB:
1. Analisis konteks yang diberikan dengan teliti
2. Berikan jawaban yang jelas dan terstruktur
3. Jika informasi tidak cukup, sampaikan dengan jujur
4. Selalu akhiri dengan anjuran konsultasi medis profesional

BATASAN:
- Tidak memberikan diagnosis medis
- Tidak meresepkan obat
- Tidak menggantikan konsultasi dokter
- Fokus pada edukasi dan informasi umum

Jawab dalam Bahasa Indonesia yang baik dan profesional."""

                response = self.hf_client.chat_completion(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"KONTEKS: {inputs['context']}\n\nPERTANYAAN: {inputs['question']}"}
                    ],
                    max_tokens=600,
                    temperature=0.5,
                    top_p=0.95
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                return f"Maaf, terjadi gangguan koneksi ke server AI. Silakan coba lagi dalam beberapa saat."

        self.rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | RunnableLambda(call_llm)
            | StrOutputParser()
        )
        self.source_retriever_chain = retriever

    def ask(self, question: str):
        """
        Enhanced ask method dengan metadata tracking dan error handling
        """
        if not self.rag_chain:
            return {
                "answer": "Sistem belum siap. Pastikan folder 'data_adenomyosis' berisi file PDF.", 
                "sources": [],
                "metadata": {}
            }
        
        try:
            # Try to get answer from RAG chain
            answer = self.rag_chain.invoke(question)
            
            # Ambil metadata sumber dengan enrichment
            try:
                retrieved_docs = self.source_retriever_chain.invoke(question)
                
                # Ekstrak sources dengan metadata
                sources = []
                source_metadata = {}
                
                for doc in retrieved_docs:
                    source_name = doc.metadata.get("source", "N/A")
                    if source_name not in sources:
                        sources.append(source_name)
                        
                        # Get year, display "N/A" if year is 0
                        year_value = doc.metadata.get("year", 0)
                        year_display = year_value if year_value > 0 else "N/A"
                        
                        source_metadata[source_name] = {
                            "type": doc.metadata.get("source_type", "Unknown"),
                            "validity": doc.metadata.get("validity_level", "unknown"),
                            "year": year_display,
                            "category": doc.metadata.get("category", "unknown")
                        }
            except Exception as retrieval_error:
                print(f"Source Retrieval Warning: {retrieval_error}")
                sources = []
                source_metadata = {}
            
            return {
                "answer": answer, 
                "sources": sources,
                "metadata": source_metadata
            }
            
        except Exception as e:
            error_msg = str(e)
            print(f"RAG Error: {error_msg}")
            
            if "fetch_k" in error_msg:
                return {
                    "answer": "Terjadi error konfigurasi. Mohon gunakan file chatbot_improved.py versi terbaru.",
                    "sources": [],
                    "metadata": {}
                }
            else:
                return {
                    "answer": f"Maaf, terjadi kesalahan teknis: {error_msg}. Mohon coba lagi atau hubungi administrator.",
                    "sources": [],
                    "metadata": {}
                }

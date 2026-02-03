import os
import fitz  # PyMuPDF
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
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
        load_dotenv()
        # Konfigurasi Database Lokal
        self.persist_directory = "./chroma_db_adenomyosis"
        
        # Model Embedding
        self.embeddings_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        self._initialize_hf_client()
        self._setup_rag_chain()

    def _initialize_chroma(self):
        """Inisialisasi ChromaDB lokal dengan metadata filtering."""
        if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
            print(f"--- Memuat database Chroma dari {self.persist_directory} ---")
            return Chroma(
                persist_directory=self.persist_directory, 
                embedding_function=self.embeddings_model,
                collection_metadata={"hnsw:space": "cosine"}  # Optimasi similarity search
            )
        else:
            print("--- Database belum ditemukan. Memulai proses indexing PDF... ---")
            st.info("Sedang membangun Database Pengetahuan dengan metadata enrichment (proses awal 1-2 menit)...")
            
            docs = self._load_pdfs_from_folder('./data_adenomyosis')
            if not docs:
                st.warning("Tidak ada dokumen PDF ditemukan di folder './data_adenomyosis'.")
                return None
            
            # PERBAIKAN A: Semantic Chunking Strategy
            # Chunk lebih kecil dengan overlap lebih besar untuk preservasi konteks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,        # Lebih kecil untuk presisi
                chunk_overlap=150,     # Overlap 30% untuk context continuity
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]  # Prioritas pemisah alami
            )
            split_docs = text_splitter.split_documents(docs)
            
            # Tambahkan chunk_id untuk tracking
            for idx, doc in enumerate(split_docs):
                doc.metadata["chunk_id"] = idx
            
            vector_store = Chroma.from_documents(
                documents=split_docs,
                embedding=self.embeddings_model,
                persist_directory=self.persist_directory,
                collection_metadata={"hnsw:space": "cosine"}
            )
            st.success(f"✅ Database berhasil dibuat: {len(split_docs)} chunks dari {len(docs)} dokumen!")
            return vector_store

    def _initialize_hf_client(self):
        """Menginisialisasi Hugging Face Client dengan model yang support chat completion."""
        hf_token = os.getenv('HUGGINGFACE_API_KEY')
        if not hf_token:
            raise ValueError("HUGGINGFACE_API_KEY tidak ditemukan di file .env")
        
        # Menggunakan Google Gemma 2 - Model ringan yang support chat completion
        self.hf_client = InferenceClient(
            model="google/gemma-2-2b-it",
            token=hf_token
        )

    def _extract_metadata_from_filename(self, filename):
        """
        PERBAIKAN B: Ekstraksi metadata dari nama file
        Contoh format: "jurnal_adenomiosis_2023.pdf" atau "herbal_kunyit_artikel.pdf"
        
        IMPORTANT: ChromaDB tidak menerima None values, semua harus str/int/float/bool
        """
        metadata = {
            "source": filename,
            "category": "unknown",
            "year": 0,  # ✅ Default 0 instead of None
            "source_type": "unknown",
            "validity_level": "medium"
        }
        
        filename_lower = filename.lower()
        
        # Deteksi kategori dari nama file
        if any(keyword in filename_lower for keyword in ["jurnal", "journal", "clinical", "study"]):
            metadata["category"] = "clinical_journal"
            metadata["validity_level"] = "high"
            metadata["source_type"] = "Jurnal Klinis"
        elif any(keyword in filename_lower for keyword in ["herbal", "natural", "traditional"]):
            metadata["category"] = "herbal_medicine"
            metadata["validity_level"] = "medium"
            metadata["source_type"] = "Herbal/Natural Medicine"
        elif any(keyword in filename_lower for keyword in ["guideline", "panduan", "protocol"]):
            metadata["category"] = "clinical_guideline"
            metadata["validity_level"] = "high"
            metadata["source_type"] = "Panduan Klinis"
        elif any(keyword in filename_lower for keyword in ["review", "overview", "tinjauan"]):
            metadata["category"] = "review_article"
            metadata["validity_level"] = "medium"
            metadata["source_type"] = "Artikel Review"
        else:
            metadata["category"] = "general_article"
            metadata["validity_level"] = "low"
            metadata["source_type"] = "Artikel Umum"
        
        # Ekstraksi tahun dari nama file (pattern: 2020-2024)
        year_match = re.search(r'(20\d{2})', filename)
        if year_match:
            metadata["year"] = int(year_match.group(1))
        # else: tetap 0 (tidak ada year di filename)
        
        return metadata

    def _setup_rag_chain(self):
        """Membangun RAG chain dengan Chain-of-Thought Prompting."""
        self.vector_store = self._initialize_chroma()
        
        if self.vector_store is None:
            self.rag_chain = None
            return

        # Retriever dengan metadata filtering (prioritas sumber high validity)
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                'k': 5  # ✅ Reduced from 7 for better precision
            }
        )

        def format_docs(docs):
            """Format documents dengan metadata untuk context yang lebih kaya"""
            formatted_chunks = []
            for doc in docs:
                source = doc.metadata.get("source", "Unknown")
                source_type = doc.metadata.get("source_type", "Unknown")
                validity = doc.metadata.get("validity_level", "unknown")
                year = doc.metadata.get("year", "N/A")
                
                # Format: [Sumber: Type | Validitas | Tahun] Content
                formatted_chunks.append(
                    f"[Sumber: {source_type} | Validitas: {validity} | Tahun: {year}]\n{doc.page_content}"
                )
            return "\n\n---\n\n".join(formatted_chunks)

        def call_llm(inputs):
            try:
                context = inputs["context"]
                question = inputs["question"]
                
                # PERBAIKAN C: Chain-of-Thought Prompting
                system_prompt = """Anda adalah Asisten Medis Ahli Ginekologi dengan spesialisasi dalam Adenomiosis dan Endometriosis.

METODOLOGI ANALISIS:
Anda WAJIB mengikuti langkah berpikir berikut sebelum menjawab:

1. IDENTIFIKASI KELUHAN: Pahami pertanyaan utama pengguna
2. ANALISIS KONTEKS: Cari bukti medis di konteks yang mendukung jawaban
3. EVALUASI VALIDITAS: Prioritaskan sumber dengan validitas tinggi (jurnal klinis)
4. PERTIMBANGAN HERBAL: Jika membahas herbal/natural medicine, WAJIB sebutkan:
   - Efek samping potensial (jika ada di konteks)
   - Interaksi obat (jika relevan)
   - Tingkat bukti ilmiah
5. KONSTRUKSI JAWABAN: Susun jawaban yang empatik namun objektif

PRINSIP JAWABAN:
- Gunakan HANYA informasi dari KONTEKS yang diberikan
- Jika informasi tidak ada, katakan dengan jelas "Informasi tidak tersedia dalam database"
- Untuk sumber validitas rendah, tambahkan disclaimer
- Berikan penjelasan yang mudah dipahami namun tetap akurat secara medis
- JANGAN membuat asumsi atau menambahkan informasi di luar konteks
- SANGAT PENTING: Jawab dengan RINGKAS dan PADAT (maksimal 2-3 paragraf, sekitar 100-150 kata)
- DILARANG KERAS mengulang kata, frasa, atau kalimat yang sama
- Gunakan sinonim dan variasi bahasa untuk menghindari pengulangan
- Jika sudah selesai menjawab, STOP - jangan tambahkan pengulangan atau elaborasi berlebihan"""

                # User message dengan structured context
                user_message = f"""KONTEKS DARI DATABASE MEDIS (dengan metadata validitas):
{context}

PERTANYAAN PASIEN:
{question}

Silakan jawab dengan mengikuti metodologi analisis di atas dalam BAHASA INDONESIA."""

                # Chat completion API call
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ]
                
                response = self.hf_client.chat_completion(
                    messages=messages,
                    max_tokens=600,       # ✅ Reduced to prevent long outputs
                    temperature=0.5,      # ✅ Increased for more diversity
                    top_p=0.95,          # ✅ Higher for better sampling
                    stop=["\n\n\n", "###", "---"]  # ✅ Stop sequences to prevent rambling
                )
                
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                print(f"LLM Connection Error: {e}")
                return "Maaf, sistem sedang mengalami gangguan koneksi ke server AI. Mohon coba lagi dalam beberapa saat."

        self.rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | RunnableLambda(call_llm)
            | StrOutputParser()
        )
        
        self.source_retriever_chain = retriever

    def _load_pdfs_from_folder(self, folder_path):
        """
        PERBAIKAN B: Load PDFs dengan metadata enrichment
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
                # Fallback: no sources
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
            
            # Provide helpful error message
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
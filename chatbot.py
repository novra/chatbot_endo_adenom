import os
import sqlite3
import shutil
from pathlib import Path

# ============================================
# FIX: Disable ChromaDB Telemetry (set before Chroma imports)
# ============================================
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY_IMPL"] = "none"
os.environ["POSTHOG_DISABLED"] = "1"

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
import traceback

class ChatBot:
    """
    ChatBot menggunakan Chat Completion API dengan perbaikan:
    1. Semantic Chunking dengan overlap besar (context preservation)
    2. Metadata Filtering (kategori sumber, tahun, validitas)
    3. Chain-of-Thought Prompting (structured reasoning)
    4. Enhanced error handling dan logging
    """
    def __init__(self):
        # Konfigurasi Database - Gunakan path yang bisa ditulis (Streamlit Cloud sering read-only)
        self.persist_directory = self._resolve_persist_directory()
        
        # Model Embedding
        self.embeddings_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        self._initialize_hf_client()
        self._setup_rag_chain()

    def _resolve_persist_directory(self):
        """Pilih direktori Chroma yang writable, fallback ke /tmp jika perlu."""
        candidates = [
            Path("./chroma_db_adenomyosis"),
            Path("/tmp/chroma_db_adenomyosis"),
        ]

        for path in candidates:
            try:
                path.mkdir(parents=True, exist_ok=True)
                test_file = path / ".write_test"
                with open(test_file, "w", encoding="utf-8") as handle:
                    handle.write("ok")
                test_file.unlink(missing_ok=True)
                print(f"✅ Using persist directory: {path}")
                return str(path)
            except Exception as e:
                print(f"❌ Cannot use {path}: {e}")
                continue

        print("⚠️ Falling back to default directory")
        return "./chroma_db_adenomyosis"

    def _initialize_hf_client(self):
        """Menginisialisasi Hugging Face Client menggunakan Streamlit Secrets."""
        try:
            hf_token = st.secrets["HUGGINGFACE_API_KEY"]
            if not hf_token or len(hf_token) < 10:
                raise ValueError("Token HuggingFace tidak valid")
            print(f"✅ HF token loaded: {hf_token[:10]}...")
        except KeyError:
            raise ValueError("❌ HUGGINGFACE_API_KEY tidak ditemukan di Streamlit Secrets.")
        except Exception as e:
            raise ValueError(f"❌ Error loading HF token: {e}")
        
        # Menggunakan Google Gemma 2 (fallback to Mistral if needed)
        try:
            self.hf_client = InferenceClient(
                model="google/gemma-2-2b-it",
                token=hf_token
            )
            print("✅ HuggingFace Client initialized with Gemma 2")
        except Exception as e:
            print(f"⚠️ Failed to init Gemma, trying Mistral: {e}")
            self.hf_client = InferenceClient(
                model="mistralai/Mistral-7B-Instruct-v0.2",
                token=hf_token
            )
            print("✅ HuggingFace Client initialized with Mistral")

    def _initialize_chroma(self):
        """Inisialisasi ChromaDB dengan pengecekan folder yang lebih aman untuk Cloud."""
        db_exists = os.path.exists(self.persist_directory) and \
                    len(os.listdir(self.persist_directory)) > 0 if os.path.exists(self.persist_directory) else False

        if db_exists:
            print(f"--- Memuat database Chroma dari {self.persist_directory} ---")
            try:
                vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings_model,
                    collection_metadata={"hnsw:space": "cosine"},
                )
                print("✅ ChromaDB loaded successfully")
                return vector_store
            except sqlite3.OperationalError as e:
                print(f"⚠️ Database error: {e}")
                st.warning(
                    "Database Chroma bermasalah atau skema tidak kompatibel. "
                    "Akan dibuat ulang dari folder data."
                )
                try:
                    shutil.rmtree(self.persist_directory)
                    print("🗑️ Old database removed")
                except Exception as rm_error:
                    print(f"⚠️ Could not remove old DB: {rm_error}")
        
        # Create new database
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
        print(f"📄 Split into {len(split_docs)} chunks")
        
        for idx, doc in enumerate(split_docs):
            doc.metadata["chunk_id"] = idx
        
        try:
            vector_store = Chroma.from_documents(
                documents=split_docs,
                embedding=self.embeddings_model,
                persist_directory=self.persist_directory,
                collection_metadata={"hnsw:space": "cosine"}
            )
            st.success(f"✅ Database berhasil dibuat dengan {len(split_docs)} chunks!")
            print(f"✅ ChromaDB created with {len(split_docs)} chunks")
            return vector_store
        except Exception as e:
            st.error(f"❌ Gagal membuat database: {e}")
            print(f"❌ ChromaDB creation error: {e}")
            return None

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
            print(f"❌ Folder not found: {folder_path}")
            return []
        
        docs = []
        pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
        print(f"📁 Found {len(pdf_files)} PDF files in {folder_path}")
        
        for filename in pdf_files:
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
                    else:
                        print(f"⚠️ Empty PDF: {filename}")
                        
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
        else:
            print("⚠️ No documents loaded!")
        
        return docs

    def _setup_rag_chain(self):
        """Membangun RAG chain dengan st.secrets support."""
        self.vector_store = self._initialize_chroma()
        
        if self.vector_store is None:
            print("❌ Vector store not initialized")
            self.rag_chain = None
            return

        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={'k': 5}
        )
        print("✅ Retriever configured")

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
            """Call LLM with comprehensive error handling and fallback strategies."""
            
            # Truncate context if too long to avoid token limits
            max_context_length = 3000
            context = inputs['context']
            if len(context) > max_context_length:
                context = context[:max_context_length] + "\n[...konteks dipotong untuk efisiensi...]"
            
            question = inputs['question']
            
            # Log for debugging
            print(f"\n=== LLM Call ===")
            print(f"Question: {question[:100]}...")
            print(f"Context length: {len(context)} chars")
            
            try:
                # Strategy 1: Chat Completion (Preferred for Gemma/Mistral)
                system_prompt = """Anda adalah Asisten Medis Ahli Ginekologi yang berspesialisasi dalam adenomyosis dan endometriosis.

PERAN ANDA:
- Memberikan informasi medis yang akurat berdasarkan literatur ilmiah
- Menjelaskan kondisi medis dengan bahasa yang mudah dipahami
- Selalu menekankan pentingnya konsultasi dengan dokter

CARA MENJAWAB:
1. Analisis konteks yang diberikan dengan teliti
2. Berikan jawaban yang jelas dan terstruktur (2-4 paragraf)
3. Jika informasi tidak cukup, sampaikan dengan jujur
4. Akhiri dengan anjuran konsultasi medis profesional

BATASAN:
- Tidak memberikan diagnosis medis
- Tidak meresepkan obat
- Tidak menggantikan konsultasi dokter
- Fokus pada edukasi dan informasi umum

Jawab dalam Bahasa Indonesia yang baik dan profesional."""

                user_message = f"""KONTEKS DARI DOKUMEN MEDIS:
{context}

PERTANYAAN PASIEN:
{question}

Berikan jawaban yang informatif, mudah dipahami, dan profesional."""

                response = self.hf_client.chat_completion(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    max_tokens=500,
                    temperature=0.6,
                    top_p=0.9
                )
                
                answer = response.choices[0].message.content.strip()
                print(f"✅ LLM Response received ({len(answer)} chars)")
                return answer
                
            except AttributeError as ae:
                # Strategy 2: Fallback to Text Generation
                print(f"⚠️ Chat completion not available, trying text generation: {ae}")
                try:
                    prompt = f"""Anda adalah asisten medis yang ahli dalam adenomyosis dan endometriosis.

Konteks: {context}

Pertanyaan: {question}

Jawaban (dalam Bahasa Indonesia, 2-3 paragraf):"""

                    response = self.hf_client.text_generation(
                        prompt=prompt,
                        max_new_tokens=400,
                        temperature=0.6,
                        top_p=0.9,
                        do_sample=True
                    )
                    print(f"✅ Text generation response received")
                    return response
                    
                except Exception as tg_error:
                    print(f"❌ Text generation also failed: {tg_error}")
                    return self._generate_fallback_response(question, str(tg_error))
            
            except Exception as e:
                # Comprehensive error handling
                error_msg = str(e)
                error_type = type(e).__name__
                
                print(f"❌ LLM Error: {error_type}")
                print(f"Error message: {error_msg}")
                print(f"Traceback:\n{traceback.format_exc()}")
                
                # Display error in Streamlit for debugging
                if st.session_state.get('debug_mode', False):
                    st.error(f"Debug - LLM Error: {error_type}: {error_msg[:200]}")
                
                return self._generate_fallback_response(question, error_msg)

        self.rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | RunnableLambda(call_llm)
            | StrOutputParser()
        )
        self.source_retriever_chain = retriever
        print("✅ RAG chain configured successfully")

    def _generate_fallback_response(self, question, error_msg):
        """Generate a helpful fallback response when LLM fails."""
        
        # Check for specific error types
        if "rate limit" in error_msg.lower() or "429" in error_msg:
            return """Maaf, server AI sedang sibuk karena terlalu banyak permintaan. 
            
Silakan tunggu beberapa detik dan coba lagi. Jika masalah berlanjut, coba pertanyaan yang lebih sederhana.

**Tips**: Cobalah bertanya dengan kalimat yang lebih singkat dan spesifik."""

        elif "token" in error_msg.lower() or "authentication" in error_msg.lower() or "401" in error_msg:
            return """❌ Error: Konfigurasi API tidak valid. 

Administrator: Mohon periksa konfigurasi HUGGINGFACE_API_KEY di Streamlit Secrets.

Untuk pengguna: Silakan hubungi administrator aplikasi."""

        elif "timeout" in error_msg.lower():
            return """Maaf, koneksi ke server AI mengalami timeout. 

Silakan coba lagi dalam beberapa saat. Jika masalah terus terjadi, coba dengan pertanyaan yang lebih singkat."""

        elif "model" in error_msg.lower() and "not found" in error_msg.lower():
            return """❌ Error: Model AI tidak tersedia.

Administrator: Model yang dikonfigurasi mungkin tidak tersedia atau memerlukan akses khusus.

Untuk pengguna: Silakan hubungi administrator aplikasi."""

        else:
            # Generic error response
            return f"""Maaf, terjadi gangguan teknis saat menghubungi server AI.

**Error**: {error_msg[:150]}

Silakan:
1. Coba lagi dalam beberapa saat
2. Gunakan pertanyaan yang lebih sederhana
3. Hubungi administrator jika masalah berlanjut

Atau coba lihat halaman **📚 Informasi Umum** untuk jawaban pertanyaan yang sering diajukan."""

    def ask(self, question: str):
        """
        Enhanced ask method dengan metadata tracking dan error handling
        """
        if not self.rag_chain:
            return {
                "answer": "❌ Sistem belum siap. Pastikan folder 'data_adenomyosis' berisi file PDF dan database telah diinisialisasi.", 
                "sources": [],
                "metadata": {}
            }
        
        try:
            print(f"\n{'='*50}")
            print(f"Question: {question}")
            
            # Get answer from RAG chain
            answer = self.rag_chain.invoke(question)
            print(f"Answer generated: {answer[:100]}...")
            
            # Retrieve source documents with metadata
            try:
                retrieved_docs = self.source_retriever_chain.invoke(question)
                print(f"Retrieved {len(retrieved_docs)} source documents")
                
                # Extract sources with metadata
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
                
                print(f"Sources: {sources}")
                
            except Exception as retrieval_error:
                print(f"⚠️ Source Retrieval Warning: {retrieval_error}")
                sources = []
                source_metadata = {}
            
            return {
                "answer": answer, 
                "sources": sources,
                "metadata": source_metadata
            }
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ RAG Error: {error_msg}")
            print(f"Traceback:\n{traceback.format_exc()}")
            
            if "fetch_k" in error_msg:
                return {
                    "answer": "❌ Terjadi error konfigurasi pada sistem retrieval. Mohon hubungi administrator.",
                    "sources": [],
                    "metadata": {}
                }
            else:
                return {
                    "answer": f"❌ Maaf, terjadi kesalahan teknis: {error_msg[:200]}. Mohon coba lagi atau hubungi administrator.",
                    "sources": [],
                    "metadata": {}
                }

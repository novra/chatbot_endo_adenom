import os
import sqlite3
import shutil
from pathlib import Path
import re
from datetime import datetime
import traceback

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

class ChatBot:
    """
    ChatBot menggunakan HuggingFace Serverless Inference dengan perbaikan:
    1. Semantic Chunking dengan overlap besar (context preservation)
    2. Metadata Filtering (kategori sumber, tahun, validitas)
    3. Enhanced error handling dan logging
    4. Proper HF token authentication
    5. Dynamic Multiple model fallback strategy (FIXED)
    """
    def __init__(self):
        # Konfigurasi Database
        self.persist_directory = self._resolve_persist_directory()
        
        # IMPORTANT: Initialize HF client FIRST to set environment variables
        self._initialize_hf_client()
        
        # Model Embedding - akan menggunakan token dari environment variable
        self.embeddings_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("✅ Embedding model initialized")
        
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
        """Menginisialisasi Hugging Face Client dengan fallback models yang kuat."""
        try:
            # Dukung HF_TOKEN atau HUGGINGFACE_API_KEY
            hf_token = st.secrets.get("HUGGINGFACE_API_KEY") or st.secrets.get("HF_TOKEN")
            if not hf_token or len(hf_token) < 10:
                raise ValueError("Token HuggingFace tidak valid atau tidak ditemukan")
            
            # SET ENVIRONMENT VARIABLES for HuggingFace Hub
            os.environ["HF_TOKEN"] = hf_token
            os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
            
            print(f"✅ HF token set in environment")
            print(f"Token preview: {hf_token[:15]}... (length: {len(hf_token)})")
            
        except Exception as e:
            raise ValueError(f"❌ Error loading HF token: {e}")
        
        # Initialize InferenceClient
        self.hf_client = InferenceClient(token=hf_token, timeout=60)
        print("✅ InferenceClient initialized for serverless inference")
        
        # FIX: Try models in order (Memprioritaskan 3.1 dan Mistral)
        models_to_try = [
            "Qwen/Qwen2.5-7B-Instruct",          # Sangat stabil dan cerdas berbahasa Indonesia
            "google/gemma-2-9b-it",              # Native support dari Google di ekosistem HF
            "microsoft/Phi-3.5-mini-instruct",   # Ringan dan hampir selalu tersedia
            "meta-llama/Llama-3.2-1B-Instruct"   # Versi Llama terkecil yang lebih diizinkan
        ]
        
        self.model_name = None
        
        for model_name in models_to_try:
            try:
                print(f"🔄 Testing model: {model_name}")
                # Test the model with a simple query
                self.hf_client.chat_completion(
                    messages=[{"role": "user", "content": "Hi"}],
                    model=model_name,
                    max_tokens=3
                )
                
                # If successful, use this model
                self.model_name = model_name
                print(f"✅ Successfully connected to: {model_name}")
                return
                
            except Exception as e:
                error_str = str(e).lower()
                print(f"⚠️ {model_name} test failed: {str(e)[:100]}")
                
                # JIKA model loading, kita tetap pilih model ini karena sebentar lagi akan siap
                if "loading" in error_str or "503" in error_str or "unavailable" in error_str:
                    print(f"⏳ {model_name} is loading, will use on first query")
                    self.model_name = model_name
                    return
                
                # JIKA model tidak disupport, biarkan loop berjalan ke model berikutnya
                continue
        
        # Fallback terakhir jika semua gagal tapi tidak crash
        if not self.model_name:
            print("⚠️ No model test successful, defaulting to Qwen")
            self.model_name = "Qwen/Qwen2.5-7B-Instruct"

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
                st.warning("Database Chroma bermasalah atau skema tidak kompatibel. Akan dibuat ulang dari folder data.")
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
        
        # Extract year from filename
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
        """Load PDFs dengan metadata enrichment."""
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
                        metadata = self._extract_metadata_from_filename(filename)
                        metadata["page_count"] = len(doc)
                        metadata["indexed_at"] = datetime.now().isoformat()
                        
                        # Filter None values and Ensure proper types
                        metadata = {k: v for k, v in metadata.items() if v is not None}
                        for key, value in metadata.items():
                            if not isinstance(value, (str, int, float, bool)):
                                metadata[key] = str(value)
                        
                        docs.append(Document(page_content=text, metadata=metadata))
                        print(f"✓ Loaded: {filename} | Category: {metadata['category']}")
                    else:
                        print(f"⚠️ Empty PDF: {filename}")
                        
            except Exception as e: 
                print(f"❌ Failed to process {filename}: {e}")
        
        # Summary
        if docs:
            categories = {}
            for doc in docs:
                cat = doc.metadata.get("category", "unknown")
                categories[cat] = categories.get(cat, 0) + 1
            
            print(f"\n📊 DOCUMENT SUMMARY:")
            print(f"Total: {len(docs)} documents")
            for cat, count in categories.items():
                print(f"  - {cat}: {count}")
        else:
            print("⚠️ No documents loaded!")
        
        return docs

    def _setup_rag_chain(self):
        """Membangun RAG chain dengan serverless inference."""
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
            """Call LLM using serverless inference with dynamic model parameter."""
            # Truncate context
            max_context_length = 1500
            context = inputs['context'][:max_context_length] if len(inputs['context']) > max_context_length else inputs['context']
            question = inputs['question']
            
            print(f"\n=== Calling {self.model_name} ===")
            print(f"Question: {question[:80]}...")
            print(f"Context: {len(context)} chars")
            
            try:
                # Prepare messages
                messages = [
                    {
                        "role": "system",
                        "content": "Anda adalah asisten medis yang ahli dalam adenomyosis dan endometriosis. Jawab dalam Bahasa Indonesia dengan jelas dan profesional."
                    },
                    {
                        "role": "user",
                        "content": f"Berdasarkan informasi medis berikut:\n\n{context}\n\nPertanyaan pasien: {question}\n\nJawab dalam 2-3 paragraf yang mudah dipahami. Akhiri dengan anjuran untuk konsultasi dokter spesialis."
                    }
                ]
                
                print("Sending to HF Serverless API...")
                
                # Call with model parameter
                response = self.hf_client.chat_completion(
                    messages=messages,
                    model=self.model_name,
                    max_tokens=400,
                    temperature=0.7,
                    top_p=0.9
                )
                
                # Extract answer
                answer = response.choices[0].message.content.strip()
                print(f"✅ Response received: {len(answer)} chars")
                return answer
                
            except Exception as e:
                error_type = type(e).__name__
                error_msg = str(e)
                
                print(f"❌ LLM Error: {error_type}")
                print(f"Error details: {error_msg[:200]}")
                print(f"Full traceback:\n{traceback.format_exc()}")
                
                # Show in debug mode
                if st.session_state.get('debug_mode', False):
                    st.error(f"🐛 DEBUG ERROR:\nType: {error_type}\nMessage: {error_msg[:400]}\nModel: {self.model_name}")
                
                return self._generate_fallback_response(question, error_msg)

        self.rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | RunnableLambda(call_llm)
            | StrOutputParser()
        )
        self.source_retriever_chain = retriever
        print("✅ RAG chain configured successfully")

    def _generate_fallback_response(self, question, error_msg):
        """Generate helpful fallback response when LLM fails."""
        error_lower = error_msg.lower()
        
        if "rate" in error_lower or "429" in error_msg:
            return "⏳ **Server AI sedang sibuk** (rate limit).\n\n**Solusi:**\n1. Tunggu 1-2 menit\n2. Coba lagi dengan pertanyaan lebih singkat\n3. Server gratis memiliki batasan request per jam"

        elif "401" in error_msg or "403" in error_msg or "token" in error_lower or "authentication" in error_lower:
            return f"🔑 **Error Autentikasi**\n\n**Admin:** Periksa HUGGINGFACE_API_KEY di Streamlit Secrets\n- Token harus valid\n\n**Error:** {error_msg[:150]}"

        elif "503" in error_msg or "loading" in error_lower or "unavailable" in error_lower:
            return "⚙️ **Model sedang loading** (cold start)\n\nModel sedang di-load ke server oleh Hugging Face.\n\n**Solusi:**\n1. Tunggu 30-60 detik\n2. Coba lagi\n3. Biasanya berhasil pada percobaan kedua"

        elif "timeout" in error_lower:
            return "⏱️ **Timeout**\n\n**Solusi:**\n1. Coba lagi (server mungkin sibuk)\n2. Gunakan pertanyaan lebih singkat\n3. Periksa koneksi internet"

        elif "model" in error_lower and ("not found" in error_lower or "not supported" in error_lower):
            return f"🤖 **Model tidak tersedia**\n\n**Error:** {error_msg[:200]}\n\n**Admin:** Model mungkin tidak didukung di infrastruktur serverless saat ini. Sistem telah mencoba mengganti model otomatis tetapi gagal."

        else:
            return f"❌ **Gangguan teknis**\n\n**Error:** {error_msg[:250]}\n\n**Solusi:** Coba lagi dalam beberapa saat. Hubungi administrator jika masalah berlanjut."

    def ask(self, question: str):
        """Ask question and get answer with sources."""
        if not self.rag_chain:
            return {
                "answer": "❌ Sistem belum siap. Database belum diinisialisasi.", 
                "sources": [],
                "metadata": {}
            }
        
        try:
            print(f"\n{'='*50}")
            print(f"Question: {question}")
            
            answer = self.rag_chain.invoke(question)
            print(f"Answer generated: {answer[:100]}...")
            
            try:
                retrieved_docs = self.source_retriever_chain.invoke(question)
                print(f"Retrieved {len(retrieved_docs)} source documents")
                
                sources = []
                source_metadata = {}
                
                for doc in retrieved_docs:
                    source_name = doc.metadata.get("source", "N/A")
                    if source_name not in sources:
                        sources.append(source_name)
                        
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
                print(f"⚠️ Source retrieval warning: {retrieval_error}")
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
            
            return {
                "answer": f"❌ Error sistem: {error_msg[:200]}. Silakan coba lagi atau hubungi administrator.",
                "sources": [],
                "metadata": {}
            }

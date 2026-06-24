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
from openai import OpenAI

class ChatBot:
    """
    ChatBot menggunakan HuggingFace Serverless Inference dengan perbaikan:
    1. Semantic Chunking dengan overlap besar (context preservation)
    2. Metadata Filtering (kategori sumber, tahun, validitas)
    3. Enhanced error handling dan logging
    4. Proper HF token authentication
    5. Dynamic Multiple model fallback strategy (FIXED)
    """
    GUARDRAIL_VERSION = "medical-scope-guardrail-v8-hf-router"
    HF_ROUTER_BASE_URL = "https://router.huggingface.co/v1"

    SCOPE_REJECTION_MESSAGE = (
        "Maaf, saya hanya dapat menjawab pertanyaan seputar endometriosis, "
        "adenomiosis/adenomyosis, serta gejala, diagnosis, perawatan, "
        "pengobatan, risiko, kesuburan, kehamilan, dan pencegahannya. "
        "Silakan ajukan pertanyaan dalam ruang lingkup tersebut. Untuk keluhan "
        "medis yang mendesak, segera hubungi dokter atau fasilitas kesehatan terdekat."
    )

    CONDITION_SCOPE_KEYWORDS = (
        "adenomyosis", "adenomiosis", "adenom", "endometriosis",
        "endometrioma", "kista coklat",
    )

    GYNECOLOGY_CONTEXT_KEYWORDS = (
        "rahim", "uterus", "endometrium", "miometrium", "panggul", "pelvis",
        "ovarium", "indung telur", "tuba", "haid", "menstruasi", "mens",
        "nyeri haid", "dismenore", "perdarahan menstruasi", "menorrhagia",
        "nyeri panggul", "infertil", "infertilitas", "kesuburan", "hamil",
        "kehamilan", "keguguran", "histerektomi", "laparoskopi", "iud", "gnrh",
    )

    CARE_SCOPE_KEYWORDS = (
        "gejala", "tanda", "penyebab", "risiko", "faktor risiko", "diagnosis",
        "diagnosa", "pemeriksaan", "skrining", "tes", "terapi", "pengobatan",
        "obat", "perawatan", "penanganan", "operasi", "pencegahan", "cegah",
        "mencegah", "komplikasi", "prognosis", "fertilitas", "kesuburan",
        "kehamilan", "nyeri", "sakit", "perdarahan", "diet", "gaya hidup",
        "olahraga", "kontrol", "konsultasi",
    )

    HERBAL_INTENT_KEYWORDS = (
        "herbal", "alami", "tradisional", "jamu", "rempah", "tanaman obat",
        "fitoterapi", "suplemen", "komplementer", "non farmakologis",
        "non-farmakologis",
    )

    FOLLOW_UP_PATTERNS = (
        r"\bapa\s+itu\s+(.+?)(?:\?|$)",
        r"\bapa\s+maksud(?:nya)?\s+(.+?)(?:\?|$)",
        r"\bjelaskan\s+(.+?)(?:\?|$)",
        r"\bmaksud\s+(.+?)\s+apa(?:\?|$)",
    )

    HERBAL_TERM_ALIASES = {
        "phaleria macrocarpa": "mahkota dewa",
        "mahkota_dewa": "mahkota dewa",
        "curcuma longa": "kunyit",
        "curcumin": "kurkumin/kunyit",
        "zingiber officinale": "jahe",
        "panax ginseng": "ginseng",
    }

    HERBAL_CONTEXT_KEYWORDS = (
        "herbal", "jamu", "phaleria", "macrocarpa", "mahkota", "flavonoid",
        "curcumin", "curcuma", "kunyit", "ginseng", "anti-inflammatory",
        "anti inflamasi", "antioxidant", "antioksidan", "il-17", "nyeri",
        "dismenore", "inflammation", "endometriosis",
    )

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
        configured_path = os.getenv("CHROMA_PERSIST_DIRECTORY")
        candidates = [
            Path(configured_path) if configured_path else None,
            Path("./chroma_db_adenomyosis"),
            Path("/tmp/chroma_db_adenomyosis"),
        ]

        for path in candidates:
            if path is None:
                continue
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

    def _chroma_db_needs_rebuild(self):
        """Detect Chroma SQLite schemas that are incompatible with chromadb 0.4.22."""
        sqlite_path = Path(self.persist_directory) / "chroma.sqlite3"
        if not sqlite_path.exists():
            return False

        try:
            with sqlite3.connect(str(sqlite_path)) as connection:
                columns = {
                    row[1]
                    for row in connection.execute("PRAGMA table_info(collections)")
                }
        except sqlite3.Error as error:
            print(f"Could not inspect Chroma schema: {error}")
            return True

        missing_columns = {"topic"} - columns
        if missing_columns:
            print(
                "Incompatible Chroma schema detected. "
                f"Missing columns: {', '.join(sorted(missing_columns))}"
            )
            return True

        return False

    def _archive_persist_directory(self):
        """Move an incompatible Chroma directory aside so a fresh DB can be built."""
        persist_path = Path(self.persist_directory)
        if not persist_path.exists():
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = persist_path.with_name(f"{persist_path.name}_backup_{timestamp}")

        try:
            shutil.move(str(persist_path), str(archive_path))
            print(f"Old ChromaDB archived to: {archive_path}")
            return True
        except Exception as move_error:
            print(f"Could not archive old ChromaDB: {move_error}")
            try:
                shutil.rmtree(str(persist_path))
                print("Old ChromaDB removed")
                return True
            except Exception as remove_error:
                print(f"Could not remove old ChromaDB: {remove_error}")
                fallback_path = persist_path.with_name(
                    f"{persist_path.name}_rebuilt_{timestamp}"
                )
                fallback_path.mkdir(parents=True, exist_ok=True)
                self.persist_directory = str(fallback_path)
                print(f"Using fresh ChromaDB directory instead: {fallback_path}")
                return False

    def _initialize_hf_client(self):
        """Menginisialisasi Hugging Face Client dengan fallback models yang kuat."""
        try:
            # Dukung Streamlit secrets dan eksekusi CLI/evaluasi via .env.
            secrets_token = None
            try:
                secrets_token = (
                    st.secrets.get("HUGGINGFACE_API_KEY")
                    or st.secrets.get("HF_TOKEN")
                )
            except Exception:
                secrets_token = None

            hf_token = (
                secrets_token
                or os.getenv("HUGGINGFACE_API_KEY")
                or os.getenv("HF_TOKEN")
            )
            if not hf_token or len(hf_token) < 10:
                raise ValueError("Token HuggingFace tidak valid atau tidak ditemukan")
            
            # SET ENVIRONMENT VARIABLES for HuggingFace Hub
            os.environ["HF_TOKEN"] = hf_token
            os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
            
            print("✅ HF token loaded")
            
        except Exception as e:
            raise ValueError(f"❌ Error loading HF token: {e}")
        
        self.hf_client = OpenAI(
            api_key=hf_token,
            base_url=self.HF_ROUTER_BASE_URL,
            timeout=60,
            max_retries=2,
        )
        print(f"✅ Hugging Face router client initialized: {self.HF_ROUTER_BASE_URL}")
        
        # FIX: Try models in order (Memprioritaskan 3.1 dan Mistral)
        models_to_try = [
            "Qwen/Qwen2.5-7B-Instruct:fastest",
            "google/gemma-2-9b-it:fastest",
            "microsoft/Phi-3.5-mini-instruct:fastest",
            "meta-llama/Llama-3.2-1B-Instruct:fastest",
        ]
        
        self.model_name = None
        
        for model_name in models_to_try:
            try:
                print(f"🔄 Testing model: {model_name}")
                # Test the model with a simple query
                self.hf_client.chat.completions.create(
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
            self.model_name = "Qwen/Qwen2.5-7B-Instruct:fastest"

    def _initialize_chroma(self):
        """Inisialisasi ChromaDB dengan pengecekan folder yang lebih aman untuk Cloud."""
        db_exists = os.path.exists(self.persist_directory) and \
                    len(os.listdir(self.persist_directory)) > 0 if os.path.exists(self.persist_directory) else False

        if db_exists and self._chroma_db_needs_rebuild():
            st.warning(
                "Database Chroma lama tidak kompatibel dengan versi ChromaDB saat ini. "
                "Database akan dibuat ulang dari PDF lokal."
            )
            self._archive_persist_directory()
            db_exists = False

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
            except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
                # Handle database schema errors (e.g., "no such column: collections.topic")
                error_msg = str(e).lower()
                if "no such column" in error_msg or "database is locked" in error_msg:
                    print(f"⚠️ Database schema mismatch atau versi ChromaDB tidak kompatibel: {e}")
                    st.warning(
                        "⚠️ Database schema tidak kompatibel (mungkin ada perubahan versi ChromaDB).\n"
                        "Database lama akan dihapus dan dibuat ulang. Ini memerlukan waktu 1-2 menit..."
                    )
                    try:
                        shutil.rmtree(self.persist_directory)
                        print("🗑️ Old database removed due to schema incompatibility")
                    except Exception as rm_error:
                        print(f"⚠️ Could not remove old DB: {rm_error}")
                else:
                    print(f"⚠️ Database error: {e}")
                    st.warning("Database Chroma bermasalah. Akan dibuat ulang dari folder data.")
                    try:
                        shutil.rmtree(self.persist_directory)
                        print("🗑️ Old database removed")
                    except Exception as rm_error:
                        print(f"⚠️ Could not remove old DB: {rm_error}")
            except Exception as e:
                # Catch any other unexpected errors
                error_msg = str(e)
                print(f"⚠️ Unexpected error loading ChromaDB: {error_msg}")
                if "no such column" in error_msg.lower() or "schema" in error_msg.lower():
                    st.warning(
                        "⚠️ Database schema error. Menghapus database lama dan membuat yang baru...\n"
                        "(Ini memerlukan waktu 1-2 menit)"
                    )
                    try:
                        shutil.rmtree(self.persist_directory)
                        print("🗑️ Old database removed due to error")
                    except Exception as rm_error:
                        print(f"⚠️ Could not remove old DB: {rm_error}")
                else:
                    st.warning(f"Error loading database: {error_msg}")
        
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

    def _is_herbal_intent(self, question: str) -> bool:
        normalized_question = self._normalize_question(question)
        return self._contains_any_keyword(normalized_question, self.HERBAL_INTENT_KEYWORDS)

    def _build_retrieval_query(self, question: str) -> str:
        if not self._is_herbal_intent(question):
            return question

        normalized_question = self._normalize_question(question)
        condition_terms = []
        if "endometriosis" in normalized_question:
            condition_terms.append("endometriosis")
        if "adenomyosis" in normalized_question or "adenomiosis" in normalized_question:
            condition_terms.append("adenomyosis adenomiosis")
        if not condition_terms:
            condition_terms.append("endometriosis adenomyosis")

        return (
            f"{question} herbal alami tradisional jamu fitoterapi suplemen "
            "anti inflamasi pereda nyeri nyeri haid dismenore keseimbangan hormon "
            "mahkota dewa phaleria macrocarpa flavonoid il-17a anti-inflammatory "
            f"antioxidant {' '.join(condition_terms)}"
        )

    def _build_contextual_question(self, question: str, chat_history=None) -> str:
        if not self._is_contextual_follow_up(question, chat_history):
            return question

        term = self._extract_follow_up_term(question)
        history_text = self._history_text(chat_history)
        context_excerpt = history_text[-900:]
        return (
            f"{question}\n\n"
            f"Konteks percakapan sebelumnya menyebut istilah '{term}' dalam pembahasan "
            f"endometriosis/adenomyosis dan herbal/komplementer. Ringkasan konteks: "
            f"{context_excerpt}"
        )

    def _herbal_alias_note(self, text: str, source_name: str = "") -> str:
        searchable_text = self._normalize_question(f"{source_name} {text}")
        aliases = []

        for source_term, indonesian_term in self.HERBAL_TERM_ALIASES.items():
            if source_term in searchable_text and indonesian_term not in aliases:
                aliases.append(indonesian_term)

        if not aliases:
            return ""

        alias_text = ", ".join(aliases)
        return f"Catatan istilah herbal: {alias_text}. Gunakan nama Indonesia ini saat menjawab."

    def _doc_context_priority(self, doc, question: str) -> int:
        source = doc.metadata.get("source", "")
        searchable_text = self._normalize_question(f"{source} {doc.page_content}")
        score = 0

        if self._is_herbal_intent(question):
            score += sum(3 for keyword in self.HERBAL_CONTEXT_KEYWORDS if keyword in searchable_text)
            if "mahkota" in searchable_text or "phaleria macrocarpa" in searchable_text:
                score += 20
            if "endometriosis" in searchable_text:
                score += 8
            if "adenomyosis" in searchable_text and "endometriosis" not in searchable_text:
                score -= 4
            if "operative" in searchable_text or "surgery" in searchable_text:
                score -= 8

        return score

    def _trim_doc_text(self, text: str, question: str, max_chars: int = 900) -> str:
        compact_text = re.sub(r"\s+", " ", text).strip()
        if len(compact_text) <= max_chars:
            return compact_text

        if self._is_herbal_intent(question):
            lowered = compact_text.lower()
            keyword_positions = [
                lowered.find(keyword)
                for keyword in self.HERBAL_CONTEXT_KEYWORDS
                if lowered.find(keyword) >= 0
            ]
            if keyword_positions:
                start = max(0, min(keyword_positions) - 250)
                return compact_text[start:start + max_chars].strip()

        return compact_text[:max_chars].strip()

    def _setup_rag_chain(self):
        """Membangun RAG chain dengan serverless inference."""
        self.vector_store = self._initialize_chroma()
        
        if self.vector_store is None:
            print("❌ Vector store not initialized")
            self.rag_chain = None
            return

        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={'k': 8}
        )
        print("✅ Retriever configured")

        def format_docs(docs, question):
            formatted_chunks = []
            sorted_docs = sorted(
                docs,
                key=lambda doc: self._doc_context_priority(doc, question),
                reverse=True,
            )

            for doc in sorted_docs:
                source = doc.metadata.get("source", "Unknown")
                source_type = doc.metadata.get("source_type", "Unknown")
                validity = doc.metadata.get("validity_level", "unknown")
                year = doc.metadata.get("year", "N/A")
                category = doc.metadata.get("category", "unknown")
                alias_note = self._herbal_alias_note(doc.page_content, source)
                alias_line = f"\n{alias_note}" if alias_note else ""
                doc_text = self._trim_doc_text(doc.page_content, question)
                formatted_chunks.append(
                    f"[File: {source} | Sumber: {source_type} | Validitas: {validity} | "
                    f"Tahun: {year} | Kategori: {category}]{alias_line}\n{doc_text}"
                )
            return "\n\n---\n\n".join(formatted_chunks)

        def retrieve_and_format(question):
            retrieval_query = self._build_retrieval_query(question)
            docs = retriever.invoke(retrieval_query)
            print("\n=== Retrieved context chunks ===")
            for idx, doc in enumerate(docs, 1):
                source = doc.metadata.get("source", "Unknown")
                chunk_id = doc.metadata.get("chunk_id", "N/A")
                priority = self._doc_context_priority(doc, question)
                preview = self._trim_doc_text(doc.page_content, question, max_chars=180)
                print(f"{idx}. {source} | chunk={chunk_id} | priority={priority} | {preview}")
            return format_docs(docs, question)

        def looks_truncated(answer):
            if not answer:
                return True

            stripped_answer = answer.rstrip()
            if stripped_answer.endswith((".", "!", "?", ")", "]")):
                return False

            trailing_words = (
                "dan", "atau", "dengan", "untuk", "dalam", "pada", "sebagai",
                "karena", "namun", "meskipun", "efek", "perbedaan", "meliputi",
                "seperti", "antara", "yang", "dapat",
            )
            last_words = stripped_answer.lower().split()[-4:]
            return True if not last_words else last_words[-1].strip(",:;") in trailing_words

        def request_completion_continuation(messages, partial_answer):
            continuation_messages = messages + [
                {
                    "role": "assistant",
                    "content": partial_answer,
                },
                {
                    "role": "user",
                    "content": (
                        "Jawaban sebelumnya terpotong. Lanjutkan langsung dari bagian terakhir "
                        "tanpa mengulang bagian awal, selesaikan kalimat yang terpotong, lalu "
                        "akhiri dengan kesimpulan dan anjuran konsultasi dokter spesialis."
                    ),
                },
            ]

            response = self.hf_client.chat.completions.create(
                messages=continuation_messages,
                model=self.model_name,
                max_tokens=450,
                temperature=0.4,
                top_p=0.9
            )
            return response.choices[0].message.content.strip()

        def call_llm(inputs):
            """Call LLM using serverless inference with dynamic model parameter."""
            # Truncate context
            max_context_length = 6500
            context = inputs['context'][:max_context_length] if len(inputs['context']) > max_context_length else inputs['context']
            question = inputs['question']
            herbal_instruction = ""
            if self._is_herbal_intent(question):
                herbal_instruction = (
                    "Pertanyaan pengguna meminta pendekatan herbal/komplementer. "
                    "Prioritaskan pembahasan dukungan herbal atau alami untuk membantu "
                    "gejala seperti inflamasi, nyeri, dan keluhan terkait hormon jika "
                    "didukung konteks. Jangan menjadikan obat farmakologis sebagai fokus "
                    "utama; sebutkan terapi medis standar hanya singkat sebagai batas "
                    "keamanan, bukan sebagai jawaban utama. Jelaskan bahwa herbal tidak "
                    "boleh diposisikan sebagai pengganti diagnosis atau terapi dokter. "
                    "Jika beberapa sumber herbal relevan tersedia, sebutkan masing-masing "
                    "herbal, nama ilmiah bila ada, mekanisme/gejala yang dituju, dan batas "
                    "buktinya."
                )
            follow_up_instruction = ""
            if self._extract_follow_up_term(question):
                follow_up_instruction = (
                    "Pertanyaan ini mungkin meminta klarifikasi istilah dari percakapan sebelumnya. "
                    "Jelaskan istilah tersebut hanya berdasarkan konteks yang tersedia. Jika istilah "
                    "tidak dikenal secara medis/herbal atau tampak salah tulis, katakan dengan jelas "
                    "bahwa istilah tersebut tidak dapat dipastikan, lalu minta pengguna mengonfirmasi "
                    "nama herbal/istilah yang dimaksud."
                )
            
            print(f"\n=== Calling {self.model_name} ===")
            print(f"Question: {question[:80]}...")
            print(f"Context: {len(context)} chars")
            
            try:
                # Prepare messages
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "Anda adalah asisten edukasi medis yang hanya membahas adenomyosis "
                            "dan endometriosis. Jawab dalam Bahasa Indonesia dengan jelas dan "
                            "profesional. Jangan menebak atau menyatakan bahwa pengguna memiliki "
                            "diagnosis, kondisi, tingkat keparahan, atau kebutuhan pengobatan tertentu "
                            "kecuali informasi itu tertulis eksplisit dalam pertanyaan. Jika informasi "
                            "tidak cukup, jelaskan secara umum dan sarankan konsultasi dokter spesialis. "
                            "Hormati intent pengguna: jika pengguna bertanya tentang herbal atau pendekatan "
                            "alami, fokuskan jawaban pada pilihan herbal/komplementer, manfaat potensial "
                            "untuk gejala, batas bukti, dan keamanan. Jika pengguna menanyakan istilah "
                            "yang muncul pada jawaban sebelumnya tetapi istilah itu tidak jelas atau tidak "
                            "didukung konteks sumber, akui ketidakpastian dan jelaskan kemungkinan salah "
                            "tulis/istilah tidak umum; jangan membuat klaim manfaat baru. Jika konteks "
                            "memuat catatan istilah herbal atau nama ilmiah, gunakan nama Indonesia dari "
                            "catatan tersebut. Jangan menerjemahkan nama tanaman secara bebas dan jangan "
                            "menciptakan nama herbal baru."
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Berdasarkan informasi medis berikut:\n\n{context}\n\n"
                            f"Pertanyaan pengguna: {question}\n\n"
                            f"{herbal_instruction}\n\n"
                            f"{follow_up_instruction}\n\n"
                            "Jawab hanya untuk ruang lingkup adenomyosis/endometriosis. "
                            "Jangan menganggap pengguna adalah pasien atau mendiagnosis pengguna. "
                            "Jika sumber menyebut Phaleria macrocarpa atau file mahkota_dewa, sebutkan "
                            "sebagai mahkota dewa (Phaleria macrocarpa), bukan istilah lain. "
                            "Untuk pertanyaan herbal, jawab lebih lengkap dalam beberapa poin terstruktur "
                            "berdasarkan sumber yang relevan. Untuk pertanyaan non-herbal, jawab dalam "
                            "2-3 paragraf yang mudah dipahami. Akhiri dengan anjuran "
                            "untuk konsultasi dokter spesialis."
                        )
                    }
                ]
                
                print("Sending to HF Serverless API...")
                
                # Call with model parameter
                response = self.hf_client.chat.completions.create(
                    messages=messages,
                    model=self.model_name,
                    max_tokens=1200,
                    temperature=0.6,
                    top_p=0.9
                )
                
                # Extract answer
                answer = response.choices[0].message.content.strip()
                if looks_truncated(answer):
                    print("Answer looks truncated; requesting continuation")
                    continuation = request_completion_continuation(messages, answer)
                    if continuation:
                        answer = f"{answer} {continuation}".strip()

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
            {
                "context": RunnableLambda(retrieve_and_format),
                "question": RunnablePassthrough(),
            }
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

    def _normalize_question(self, question: str) -> str:
        """Normalize user question for deterministic guardrail checks."""
        normalized = (question or "").lower()
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    def _contains_any_keyword(self, text: str, keywords) -> bool:
        return any(keyword in text for keyword in keywords)

    def _extract_follow_up_term(self, question: str) -> str:
        normalized_question = self._normalize_question(question)
        for pattern in self.FOLLOW_UP_PATTERNS:
            match = re.search(pattern, normalized_question)
            if match:
                term = match.group(1).strip(" .,:;!?")
                term = re.sub(r"^(tentang|mengenai|soal)\s+", "", term)
                return term.strip()
        return ""

    def _history_text(self, chat_history) -> str:
        if not chat_history:
            return ""

        parts = []
        for message in chat_history[-6:]:
            if not isinstance(message, dict):
                continue
            content = message.get("content", "")
            if content:
                parts.append(str(content))
        return self._normalize_question(" ".join(parts))

    def _is_contextual_follow_up(self, question: str, chat_history=None) -> bool:
        term = self._extract_follow_up_term(question)
        if len(term) < 3:
            return False

        history_text = self._history_text(chat_history)
        if not history_text or term not in history_text:
            return False

        return self._is_in_medical_scope(history_text)

    def _is_in_medical_scope(self, question: str) -> bool:
        """
        Guardrail: only allow adenomyosis/endometriosis questions and closely
        related gynecologic care, treatment, diagnosis, and prevention topics.
        """
        normalized_question = self._normalize_question(question)

        if len(normalized_question) < 3:
            return False

        has_condition = self._contains_any_keyword(
            normalized_question,
            self.CONDITION_SCOPE_KEYWORDS,
        )
        has_gynecology_context = self._contains_any_keyword(
            normalized_question,
            self.GYNECOLOGY_CONTEXT_KEYWORDS,
        )
        has_care_context = self._contains_any_keyword(
            normalized_question,
            self.CARE_SCOPE_KEYWORDS,
        )

        return has_condition or (has_gynecology_context and has_care_context)

    def ask(self, question: str, chat_history=None):
        """Ask question and get answer with sources."""
        contextual_question = self._build_contextual_question(question, chat_history)

        if not self._is_in_medical_scope(question) and contextual_question == question:
            print("Guardrail blocked out-of-scope question")
            return {
                "answer": self.SCOPE_REJECTION_MESSAGE,
                "sources": [],
                "metadata": {
                    "guardrail": {
                        "blocked": True,
                        "reason": "out_of_scope"
                    }
                }
            }

        if not self.rag_chain:
            return {
                "answer": "❌ Sistem belum siap. Database belum diinisialisasi.",
                "sources": [],
                "metadata": {}
            }

        try:
            print(f"\n{'='*50}")
            print(f"Question: {question}")

            answer = self.rag_chain.invoke(contextual_question)
            print(f"Answer generated: {answer[:100]}...")

            try:
                retrieved_docs = self.source_retriever_chain.invoke(
                    self._build_retrieval_query(contextual_question)
                )
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

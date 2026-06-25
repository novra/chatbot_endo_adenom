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
    GUARDRAIL_VERSION = "medical-scope-guardrail-v19-specific-herbal-object"
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
        "olahraga", "kontrol", "konsultasi", "rekomendasi",
        "direkomendasikan", "manfaat", "khasiat",
    )

    HERBAL_INTENT_KEYWORDS = (
        "herbal", "alami", "tradisional", "jamu", "rempah", "tanaman obat",
        "fitoterapi", "suplemen", "komplementer", "non farmakologis",
        "non-farmakologis", "konvensional",
    )

    FOLLOW_UP_PATTERNS = (
        r"\bapa\s+itu\s+(.+?)(?:\?|$)",
        r"\bapa\s+maksud(?:nya)?\s+(.+?)(?:\?|$)",
        r"\bjelaskan\s+(.+?)(?:\?|$)",
        r"\bmaksud\s+(.+?)\s+apa(?:\?|$)",
    )

    FOLLOW_UP_REFERENCE_KEYWORDS = (
        "itu", "tersebut", "tadi", "sebelumnya", "di atas", "diatas",
        "yang tadi", "hal itu", "kondisi itu", "penyakit itu", "gejala itu",
        "pengobatannya", "terapinya", "diagnosisnya", "penyebabnya",
        "gejalanya", "risikonya", "dampaknya", "pencegahannya",
        "menggunakannya", "memakainya", "penggunaannya", "pemakaiannya",
        "keamanannya", "dosisnya", "efek sampingnya", "interaksinya",
        "yang perlu diperhatikan",
    )

    FOLLOW_UP_INTENT_KEYWORDS = (
        "lanjut", "lanjutkan", "lebih lanjut", "jelaskan lagi", "detail",
        "lebih detail", "bagaimana", "apakah", "kenapa", "mengapa",
        "berapa", "apa saja", "contohnya", "contoh", "bedanya",
        "perhatikan",
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

    HERBAL_SYMPTOM_CONTEXT_KEYWORDS = (
        "gejala", "nyeri", "sakit", "dismenore", "kram", "haid",
        "menstruasi", "perdarahan", "inflamasi", "anti inflamasi",
        "analgesik", "kunyit", "kunyit asam", "curcuma", "curcumin",
        "kurkumin", "jahe", "ginger", "ramuan",
    )

    HERBAL_ALTERNATIVE_SOURCE_KEYWORDS = (
        "kunyit", "curcuma", "curcumin", "kurkumin", "jahe", "ginger",
        "ramuan", "aplikasi_herbal", "herbal_umum", "jamu_kunyit_asam",
    )

    HERBAL_SIGNAL_KEYWORDS = (
        "herbal", "jamu", "phaleria", "macrocarpa", "mahkota", "curcumin",
        "curcuma", "kunyit", "ginseng", "tanaman obat", "fitoterapi",
        "suplemen", "komplementer",
    )

    SPECIFIC_HERBAL_OBJECTS = {
        "mahkota dewa": ("mahkota dewa", "phaleria macrocarpa", "mahkota_dewa"),
        "kunyit/kurkumin": ("kunyit", "curcuma longa", "curcumin", "kurkumin"),
        "jahe": ("jahe", "zingiber officinale", "ginger"),
        "ginseng": ("ginseng", "panax ginseng"),
    }

    RETRIEVAL_STOPWORDS = {
        "yang", "dan", "atau", "untuk", "dengan", "dalam", "pada", "dari",
        "apa", "itu", "saja", "biasanya", "sebelum", "menggunakan", "keluhan",
        "dapat", "perlu", "dokter", "terapi",
    }

    DIAGNOSIS_CONTEXT_KEYWORDS = (
        "diagnosis", "diagnostik", "pemeriksaan", "anamnesis", "usg",
        "ultrasound", "transvaginal", "mri", "histologi", "histopatologis",
        "sensitivitas", "spesifisitas",
    )

    TREATMENT_CONTEXT_KEYWORDS = (
        "tatalaksana", "penatalaksanaan", "terapi", "pengobatan",
        "medikamentosa", "hormonal", "oains", "nsaid", "progestin",
        "lng-ius", "levonorgestrel", "gnrh", "histerektomi", "operatif",
        "operasi", "pembedahan", "laparoskopi", "eksisi", "ablasi",
        "konservatif", "klinis", "guideline", "evidence", "fisioterapi",
        "terapi fisik", "pelvic floor", "rehabilitasi",
    )

    INFERTILITY_CONTEXT_KEYWORDS = (
        "infertilitas", "infertil", "subfertilitas", "fertilitas",
        "implantasi", "junctional zone", "radikal bebas", "ivf",
        "in vitro fertilization", "fertilisasi",
    )

    HERBAL_SAFETY_CONTEXT_KEYWORDS = (
        "keamanan", "efek samping", "interaksi", "dosis", "kualitas",
        "konvensional", "konsultasi", "hormonal", "kehamilan", "operasi",
        "tidak boleh", "pengganti",
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
            "google/gemma-2-9b-it:fastest",
            "Qwen/Qwen2.5-7B-Instruct:fastest",
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
        return (
            self._contains_any_keyword(normalized_question, self.HERBAL_INTENT_KEYWORDS)
            or self._specific_herbal_topic(question) != ""
        )

    def _specific_herbal_topic(self, question: str) -> str:
        normalized_question = self._normalize_question(question)
        for topic, aliases in self.SPECIFIC_HERBAL_OBJECTS.items():
            if any(alias in normalized_question for alias in aliases):
                return topic
        return ""

    def _is_treatment_intent(self, question: str) -> bool:
        normalized_question = self._normalize_question(question)
        return any(
            keyword in normalized_question
            for keyword in (
                "pengobatan", "pengobatannya", "terapi", "terapinya",
                "tatalaksana", "penanganan", "perawatan", "pembedahan",
                "operasi",
            )
        )

    def _is_herbal_safety_intent(self, question: str) -> bool:
        normalized_question = self._normalize_question(question)
        return self._is_herbal_intent(question) and any(
            keyword in normalized_question
            for keyword in (
                "ganti", "menggantikan", "pengganti", "perhatikan", "sebelum",
                "aman", "keamanan", "efek samping", "interaksi", "dosis",
            )
        )

    def _build_retrieval_query(self, question: str) -> str:
        question = self._split_contextual_question(question)[0]
        normalized_question = self._normalize_question(question)
        treatment_intent = self._is_treatment_intent(question)
        specific_herbal_topic = self._specific_herbal_topic(question)

        if not self._is_herbal_intent(question):
            if treatment_intent:
                return (
                    f"{question} tatalaksana klinis berbasis bukti terapi medis "
                    "endometriosis adenomiosis obat anti nyeri oains nsaid "
                    "terapi hormonal kontrasepsi kombinasi progestin dienogest "
                    "lng-ius levonorgestrel gnrh agonist antagonist aromatase "
                    "inhibitor terapi fisik fisioterapi panggul pelvic floor "
                    "tindakan konservatif laparoskopi eksisi ablasi adhesiolisis "
                    "adenomiomektomi embolisasi histerektomi pembedahan guideline "
                    "dokter spesialis"
                )
            return question

        is_safety_or_replacement_intent = self._is_herbal_safety_intent(question)
        condition_terms = []
        if "endometriosis" in normalized_question:
            condition_terms.append("endometriosis")
        if "adenomyosis" in normalized_question or "adenomiosis" in normalized_question:
            condition_terms.append("adenomyosis adenomiosis")
        if not condition_terms:
            condition_terms.append("endometriosis adenomyosis")

        if is_safety_or_replacement_intent:
            return (
                f"{question} herbal komplementer keamanan efek samping dosis "
                "kualitas produk interaksi obat terapi hormonal terapi konvensional "
                "tidak menggantikan diagnosis dokter konsultasi dokter batas bukti "
                f"{' '.join(condition_terms)}"
            )

        if specific_herbal_topic == "mahkota dewa":
            return (
                f"{question} mahkota dewa phaleria macrocarpa flavonoid il-17a "
                "anti-inflammatory anti inflamasi endometriosis mekanisme potensi "
                "manfaat rekomendasi batas bukti keamanan efek samping interaksi "
                f"{' '.join(condition_terms)}"
            )

        return (
            f"{question} herbal alami tradisional jamu fitoterapi suplemen "
            "komplementer konvensional pendamping "
            "penanganan gejala nyeri haid dismenore inflamasi kram menstruasi "
            "anti inflamasi anti nyeri analgesik kunyit asam curcuma longa "
            "kurkumin curcumin jahe zingiber officinale ramuan herbal jamu "
            "aplikasi herbal umum mahkota dewa phaleria macrocarpa flavonoid "
            "antioxidant phytotherapy variasi herbal "
            f"{' '.join(condition_terms)}"
        )

    def _build_contextual_question(self, question: str, chat_history=None) -> str:
        if not self._is_contextual_follow_up(question, chat_history):
            return question

        standalone_question = self._build_standalone_follow_up_question(
            question,
            chat_history,
        )
        history_text = self._history_text(chat_history)
        context_excerpt = history_text[-1200:]
        return (
            f"{standalone_question}\n\n"
            f"Pertanyaan asli pengguna: {question}\n"
            f"Riwayat ringkas yang relevan: {context_excerpt}"
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
        category = doc.metadata.get("category", "")
        question = self._split_contextual_question(question)[0]
        normalized_question = self._normalize_question(question)
        searchable_text = self._normalize_question(f"{source} {doc.page_content}")
        question_terms = {
            term
            for term in re.findall(r"[a-zA-ZÀ-ÿ0-9]+", normalized_question)
            if len(term) > 3 and term not in self.RETRIEVAL_STOPWORDS
        }
        score = 0

        score += min(18, sum(2 for term in question_terms if term in searchable_text))

        if "adenomiosis" in normalized_question or "adenomyosis" in normalized_question:
            if category == "adenomyosis":
                score += 10
            if "adenomyosis" in searchable_text or "adenomiosis" in searchable_text:
                score += 6

        if normalized_question.startswith("apa itu"):
            if any(keyword in searchable_text for keyword in ("adalah", "merupakan", "invasi jinak", "miometrium")):
                score += 12
            if any(keyword in searchable_text for keyword in ("diagnosis", "pemeriksaan", "terapi", "tatalaksana", "infertilitas", "ivf")):
                score -= 8

        if "endometriosis" in normalized_question:
            if category == "endometriosis":
                score += 12
            if "endometriosis" in searchable_text:
                score += 8

        if any(keyword in normalized_question for keyword in ("diagnosis", "didiagnosis", "pemeriksaan", "diagnosa")):
            score += sum(5 for keyword in self.DIAGNOSIS_CONTEXT_KEYWORDS if keyword in searchable_text)
            if any(keyword in searchable_text for keyword in ("fertilitas", "infertilitas", "ivf", "treatment", "tatalaksana")):
                score -= 6

        has_treatment_intent = self._is_treatment_intent(question)

        if has_treatment_intent:
            score += sum(5 for keyword in self.TREATMENT_CONTEXT_KEYWORDS if keyword in searchable_text)
            if category == "treatment":
                score += 8
            if any(keyword in searchable_text for keyword in ("guideline", "clinical", "klinis", "medikamentosa")):
                score += 8
            if any(keyword in searchable_text for keyword in ("oains", "nsaid", "hormonal", "progestin", "lng-ius", "gnrh", "histerektomi")):
                score += 10
            if any(keyword in searchable_text for keyword in ("laparoskopi", "laparoscopy", "eksisi", "excision", "ablasi", "ablation", "operative", "surgery")):
                score += 8
            if any(keyword in searchable_text for keyword in ("fisioterapi", "terapi fisik", "pelvic floor", "rehabilitasi")):
                score += 8
            if any(keyword in searchable_text for keyword in ("fertilitas", "infertilitas", "ivf")):
                score -= 5
            if not self._is_herbal_intent(question):
                if any(keyword in searchable_text for keyword in self.HERBAL_SIGNAL_KEYWORDS):
                    score -= 18
                if any(keyword in source.lower() for keyword in ("herbal", "jamu", "mahkota")):
                    score -= 20

        if any(keyword in normalized_question for keyword in ("infertil", "infertilitas", "fertilitas", "kesuburan")):
            score += sum(6 for keyword in self.INFERTILITY_CONTEXT_KEYWORDS if keyword in searchable_text)
            if category == "adenomyosis":
                score += 4

        if "perbedaan" in normalized_question and "endometriosis" in normalized_question:
            if ("adenomyosis" in searchable_text or "adenomiosis" in searchable_text) and "endometriosis" in searchable_text:
                score += 16
            if any(keyword in searchable_text for keyword in ("fertilitas", "ivf", "treatment")):
                score -= 4

        if self._is_herbal_intent(question):
            source_lower = source.lower()
            is_adjacent_topic = any(keyword in source_lower for keyword in ("kanker", "cervical", "fibroid", "miom"))
            explicit_mahkota_intent = any(
                keyword in normalized_question
                for keyword in ("mahkota", "phaleria", "macrocarpa")
            )
            symptom_herbal_intent = any(
                keyword in normalized_question
                for keyword in (
                    "gejala", "nyeri", "sakit", "dismenore", "haid",
                    "menstruasi", "perdarahan", "kram", "inflamasi",
                    "penanganan", "anti nyeri", "antiinflamasi",
                    "anti inflamasi",
                )
            )
            if any(keyword in source_lower for keyword in ("herbal", "jamu", "mahkota")) and not is_adjacent_topic:
                score += 14
            if not any(keyword in searchable_text for keyword in self.HERBAL_CONTEXT_KEYWORDS):
                score -= 20
            score += sum(3 for keyword in self.HERBAL_CONTEXT_KEYWORDS if keyword in searchable_text)
            if explicit_mahkota_intent and ("mahkota" in searchable_text or "phaleria macrocarpa" in searchable_text):
                score += 20
            elif not explicit_mahkota_intent and any(keyword in source_lower for keyword in ("mahkota", "phaleria")):
                score += 8
            if not explicit_mahkota_intent:
                score += sum(4 for keyword in self.HERBAL_SYMPTOM_CONTEXT_KEYWORDS if keyword in searchable_text)
                if any(keyword in source_lower for keyword in self.HERBAL_ALTERNATIVE_SOURCE_KEYWORDS):
                    score += 24
                if any(keyword in searchable_text for keyword in ("curcuma", "curcumin", "kurkumin", "kunyit", "jahe", "ginger")):
                    score += 12
            if symptom_herbal_intent:
                score += sum(5 for keyword in self.HERBAL_SYMPTOM_CONTEXT_KEYWORDS if keyword in searchable_text)
                if any(keyword in source_lower for keyword in self.HERBAL_ALTERNATIVE_SOURCE_KEYWORDS):
                    score += 16
            if "endometriosis" in searchable_text:
                score += 8
            if "adenomyosis" in searchable_text and "endometriosis" not in searchable_text:
                score -= 4
            if "operative" in searchable_text or "surgery" in searchable_text:
                score -= 8
            if any(keyword in normalized_question for keyword in ("ganti", "menggantikan", "pengganti", "perhatikan", "sebelum")):
                score += sum(5 for keyword in self.HERBAL_SAFETY_CONTEXT_KEYWORDS if keyword in searchable_text)
            if not any(keyword in normalized_question for keyword in ("serviks", "kanker", "fibroid", "miom")):
                if any(keyword in searchable_text for keyword in ("kanker serviks", "cervical cancer", "fibroid", "mioma")):
                    score -= 35
            if not any(keyword in normalized_question for keyword in ("infertil", "fertilitas", "kesuburan")):
                if any(keyword in searchable_text for keyword in ("infertilitas", "fertilitas", "ivf", "in vitro fertilization")):
                    score -= 10

        return score

    def _diversify_docs_by_source(self, docs, limit: int):
        selected = []
        selected_sources = set()

        for doc in docs:
            source = doc.metadata.get("source", "Unknown")
            if source in selected_sources:
                continue
            selected.append(doc)
            selected_sources.add(source)
            if len(selected) >= limit:
                return selected

        for doc in docs:
            if doc in selected:
                continue
            selected.append(doc)
            if len(selected) >= limit:
                break

        return selected

    def _retrieve_relevant_docs(self, question: str, limit: int = 3):
        question = self._split_contextual_question(question)[0]
        retrieval_query = self._build_retrieval_query(question)
        candidate_docs = self.source_retriever_chain.invoke(retrieval_query)
        ranked_docs = sorted(
            candidate_docs,
            key=lambda doc: self._doc_context_priority(doc, question),
            reverse=True,
        )
        if self._is_herbal_intent(question):
            return self._diversify_docs_by_source(ranked_docs, max(limit, 4))
        return ranked_docs[:limit]

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
            search_kwargs={'k': 24}
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
            docs = self._retrieve_relevant_docs(question)
            print("\n=== Retrieved context chunks ===")
            for idx, doc in enumerate(docs, 1):
                source = doc.metadata.get("source", "Unknown")
                chunk_id = doc.metadata.get("chunk_id", "N/A")
                priority = self._doc_context_priority(doc, question)
                preview = self._trim_doc_text(doc.page_content, question, max_chars=180)
                print(f"{idx}. {source} | chunk={chunk_id} | priority={priority} | {preview}")
            return format_docs(docs, question)

        def extract_question_and_history(question):
            parts = str(question).split("\n\n", 1)
            user_question = parts[0].strip()
            history_context = parts[1].strip() if len(parts) > 1 else ""
            return user_question, history_context

        def looks_truncated(answer):
            if not answer:
                return True

            stripped_answer = answer.rstrip()
            if stripped_answer.endswith((".", "!", "?", ")", "]")):
                return False
            if len(stripped_answer.split()) > 12:
                return True

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
            user_question, history_context = extract_question_and_history(question)
            specific_herbal_topic = self._specific_herbal_topic(user_question)
            herbal_instruction = ""
            if self._is_herbal_intent(user_question):
                if self._is_herbal_safety_intent(user_question):
                    herbal_instruction = (
                        "Pertanyaan pengguna berfokus pada keamanan herbal atau apakah herbal "
                        "dapat menggantikan terapi dokter. Jawab langsung bahwa herbal tidak "
                        "boleh menggantikan diagnosis atau terapi dokter. Fokus pada batas bukti, "
                        "keamanan, dosis, kualitas produk, efek samping, interaksi dengan obat "
                        "atau terapi hormonal, kondisi kehamilan/rencana operasi, dan perlunya "
                        "konsultasi tenaga kesehatan. Untuk pertanyaan keamanan, jangan menyebut "
                        "nama herbal spesifik kecuali pengguna menyebut nama herbal tersebut secara "
                        "eksplisit. Jangan membuat daftar herbal. Jawab ringkas dalam 1 paragraf "
                        "pembuka dan 4-6 poin checklist. Isi checklist harus mencakup: diagnosis "
                        "dan evaluasi dokter, bukti klinis terbatas, keamanan, dosis, kualitas "
                        "produk, efek samping, interaksi obat/terapi hormonal, kehamilan atau "
                        "rencana operasi, dan konsultasi tenaga kesehatan."
                    )
                else:
                    herbal_instruction = (
                        "Pertanyaan pengguna meminta pendekatan herbal/komplementer. "
                        "Fokuskan jawaban pada terapi herbal/komplementer yang relevan dari konteks, "
                        "terutama potensi untuk gejala seperti nyeri, dismenore, dan inflamasi. "
                        "Jangan menjadikan terapi klinis seperti hormonal, prosedur, atau pembedahan "
                        "sebagai isi utama pada pertanyaan ini. Terapi klinis boleh disebut hanya "
                        "singkat sebagai batas keamanan bahwa herbal/komplementer bukan pengganti "
                        "diagnosis dan terapi dokter. Jelaskan batas bukti, keamanan, efek samping, "
                        "interaksi obat/terapi hormonal, kualitas produk, dan perlunya konsultasi "
                        "tenaga kesehatan. Jika beberapa sumber herbal relevan tersedia, gunakan "
                        "variasi sumber seperti kunyit/kurkumin, kunyit asam, jahe, mahkota dewa/"
                        "Phaleria macrocarpa, atau ramuan herbal bila ada dalam konteks. Bila "
                        "mahkota dewa muncul di konteks, tetap sebutkan sebagai salah satu opsi "
                        "potensial, tetapi selalu sandingkan dengan herbal lain yang relevan."
                    )
                if specific_herbal_topic:
                    herbal_instruction += (
                        f" Pertanyaan pengguna menyebut objek herbal spesifik: {specific_herbal_topic}. "
                        "Fokuskan jawaban hanya pada objek tersebut: alasan dapat direkomendasikan, "
                        "kandungan/mekanisme yang didukung konteks, gejala yang mungkin dituju, "
                        "batas bukti, keamanan, efek samping, dan interaksi. Jangan membuat daftar "
                        "herbal lain kecuali pengguna meminta perbandingan atau alternatif."
                    )
            treatment_instruction = ""
            clinical_treatment_intent = (
                self._is_treatment_intent(user_question)
                and not self._is_herbal_intent(user_question)
            )
            if clinical_treatment_intent:
                treatment_instruction = (
                    "Untuk pertanyaan perawatan/pengobatan/terapi, jawaban utama harus spesifik "
                    "dan menyeluruh tentang tatalaksana klinis berbasis bukti. Susun dengan urutan: "
                    "(1) evaluasi dokter dan tujuan terapi; (2) obat nyeri/antiinflamasi seperti "
                    "OAINS/NSAID bila sesuai; (3) terapi hormonal, misalnya kontrasepsi hormonal, "
                    "progestin/dienogest, LNG-IUS, agonis/antagonis GnRH, dan opsi lain yang "
                    "didukung konteks; (4) terapi fisik atau rehabilitasi panggul sebagai dukungan "
                    "gejala nyeri bila relevan; (5) prosedur atau pembedahan, misalnya laparoskopi "
                    "eksisi/ablasi untuk endometriosis, tindakan konservatif pada adenomiosis, "
                    "embolisasi/adenomiomektomi bila sesuai, dan histerektomi sebagai terapi definitif "
                    "adenomiosis pada pasien yang tidak merencanakan kehamilan; (6) pertimbangan "
                    "fertilitas dan rencana hamil. Pendekatan komplementer seperti herbal hanya boleh "
                    "disebut singkat di akhir sebagai pendamping bila pengguna menanyakannya atau "
                    "bila perlu sebagai catatan keamanan, bukan sebagai bagian utama."
                )
                treatment_instruction += (
                    " Karena pengguna menanyakan perawatan/pengobatan umum dan tidak meminta herbal, "
                    "jangan membahas daftar herbal, jamu, atau suplemen. Jika menyebut komplementer, "
                    "cukup satu kalimat singkat bahwa pendekatan tersebut tidak menggantikan terapi "
                    "klinis dan perlu dikonsultasikan."
                )
            intent_instruction = ""
            if self._normalize_question(user_question).startswith("apa itu"):
                intent_instruction = (
                    "Pertanyaan ini meminta definisi. Jawab dengan definisi langsung dan gejala "
                    "utama yang paling relevan saja dalam maksimal 2 paragraf. Jangan memakai daftar "
                    "poin. Jangan memperluas ke diagnosis, pemeriksaan, infertilitas, atau terapi "
                    "kecuali diminta. Instruksi definisi ini lebih prioritas daripada instruksi format lain."
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
            if "diagnosis" in self._normalize_question(user_question) or "pemeriksaan" in self._normalize_question(user_question):
                follow_up_instruction += (
                    " Untuk pertanyaan diagnosis atau pemeriksaan adenomiosis, fokus pada anamnesis/"
                    "gejala, pemeriksaan klinis/panggul, USG transvaginal, MRI, dan histopatologi "
                    "bila relevan. Jangan menambahkan CT scan atau histeroskopi kecuali tertulis jelas "
                    "dalam konteks sumber."
                )
            if any(keyword in self._normalize_question(user_question) for keyword in ("hamil", "kehamilan", "fertilitas", "kesuburan")):
                follow_up_instruction += (
                    " Jika pengguna ingin hamil atau mempertahankan kesuburan, nyatakan jelas bahwa "
                    "histerektomi adalah terapi definitif untuk pasien yang tidak lagi menginginkan "
                    "kehamilan, sehingga tidak sesuai sebagai pilihan bagi pasien yang masih ingin hamil. "
                    "Fokus pada evaluasi dokter, kontrol gejala, terapi sementara, reseksi konservatif "
                    "terpilih, dan IVF/GnRH bila didukung konteks."
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
                            "Untuk pertanyaan perawatan, pengobatan, terapi, atau tatalaksana, "
                            "ikuti intent pengguna. Jika pengguna bertanya umum tentang perawatan "
                            "atau pengobatan, jadikan pilihan medis klinis berbasis bukti sebagai "
                            "isi utama: obat nyeri, terapi hormonal, terapi fisik/rehabilitasi panggul "
                            "bila relevan, prosedur konservatif, dan pembedahan. Jika pengguna jelas "
                            "bertanya tentang herbal, terapi konvensional/tradisional, atau pengobatan "
                            "komplementer, fokuskan jawaban pada pendekatan herbal/komplementer dan "
                            "sebutkan terapi klinis hanya singkat sebagai batas keamanan, bukan isi utama. "
                            "Jika pengguna menanyakan istilah "
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
                            f"Riwayat percakapan relevan:\n{history_context or 'Tidak ada.'}\n\n"
                            f"Pertanyaan pengguna: {user_question}\n\n"
                            f"{herbal_instruction}\n\n"
                            f"{treatment_instruction}\n\n"
                            f"{intent_instruction}\n\n"
                            f"{follow_up_instruction}\n\n"
                            "Jawab hanya untuk ruang lingkup adenomyosis/endometriosis. "
                            "Jangan menganggap pengguna adalah pasien atau mendiagnosis pengguna. "
                            "Jika sumber menyebut Phaleria macrocarpa atau file mahkota_dewa, sebutkan "
                            "sebagai mahkota dewa (Phaleria macrocarpa), bukan istilah lain, tetapi "
                            "jangan menjadikannya contoh tunggal bila pengguna tidak menanyakannya. "
                            "Untuk pertanyaan herbal/komplementer/konvensional, fokuskan jawaban pada "
                            "pendekatan tersebut sesuai intent pengguna; untuk gejala, antiinflamasi, "
                            "atau anti-nyeri, prioritaskan "
                            "sumber herbal yang membahas kunyit/kurkumin, kunyit asam, jahe, mahkota "
                            "dewa, atau ramuan herbal bila tersedia. Mahkota dewa tetap boleh muncul "
                            "sebagai salah satu opsi, tetapi jangan hanya berfokus pada mahkota dewa "
                            "kecuali pengguna menyebutnya. "
                            "Bila pertanyaan tentang keamanan/pengganti dokter, "
                            "fokuskan pada batas keamanan dan jangan memperpanjang daftar herbal. "
                            "Untuk pertanyaan non-herbal tentang perawatan/pengobatan, jawab dengan "
                            "bagian klinis yang jelas dan cukup lengkap, bukan daftar herbal. Untuk "
                            "pertanyaan non-herbal lain, jawab dalam 2-3 paragraf yang mudah dipahami. "
                            "Akhiri dengan anjuran "
                            "untuk konsultasi dokter spesialis."
                        )
                    }
                ]
                
                print("Sending to HF Serverless API...")
                
                # Call with model parameter
                max_answer_tokens = 650
                if self._is_herbal_safety_intent(user_question):
                    max_answer_tokens = 600
                elif not self._normalize_question(user_question).startswith("apa itu"):
                    max_answer_tokens = 1200

                response = self.hf_client.chat.completions.create(
                    messages=messages,
                    model=self.model_name,
                    max_tokens=max_answer_tokens,
                    temperature=0.2,
                    top_p=0.85
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

    def _split_contextual_question(self, question: str) -> tuple[str, str]:
        parts = str(question).split("\n\n", 1)
        user_question = parts[0].strip()
        history_context = parts[1].strip() if len(parts) > 1 else ""
        return user_question, history_context

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
        for message in chat_history[-20:]:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "message")).lower()
            content = message.get("content", "")
            if content:
                compact_content = self._normalize_question(str(content))[:900]
                parts.append(f"{role}: {compact_content}")
        return self._normalize_question(" ".join(parts))

    def _last_topic_from_history(self, chat_history) -> str:
        if not chat_history:
            return ""

        user_messages = [
            message for message in chat_history
            if isinstance(message, dict) and message.get("role") == "user"
        ]
        candidate_messages = user_messages[-10:] if user_messages else chat_history[-20:]

        for message in reversed(candidate_messages):
            if not isinstance(message, dict):
                continue
            content = self._normalize_question(str(message.get("content", "")))
            if not content:
                continue
            if self._contains_any_keyword(content, self.HERBAL_SIGNAL_KEYWORDS):
                if "endometriosis" in content and (
                    "adenomiosis" in content or "adenomyosis" in content
                ):
                    return "herbal untuk endometriosis atau adenomiosis"
                if "endometriosis" in content:
                    return "herbal untuk endometriosis"
                if "adenomiosis" in content or "adenomyosis" in content:
                    return "herbal untuk adenomiosis"
                return "herbal/komplementer"
            if "adenomiosis" in content or "adenomyosis" in content:
                return "adenomiosis"
            if "endometriosis" in content:
                return "endometriosis"

        return ""

    def _last_condition_from_history(self, chat_history) -> str:
        if not chat_history:
            return ""

        user_messages = [
            message for message in chat_history
            if isinstance(message, dict) and message.get("role") == "user"
        ]
        candidate_messages = user_messages[-10:] if user_messages else chat_history[-20:]

        for message in reversed(candidate_messages):
            if not isinstance(message, dict):
                continue
            content = self._normalize_question(str(message.get("content", "")))
            if not content:
                continue
            has_endometriosis = "endometriosis" in content
            has_adenomyosis = "adenomiosis" in content or "adenomyosis" in content
            if has_endometriosis and has_adenomyosis:
                return "endometriosis atau adenomiosis"
            if has_endometriosis:
                return "endometriosis"
            if has_adenomyosis:
                return "adenomiosis"

        return ""

    def _build_standalone_follow_up_question(self, question: str, chat_history=None) -> str:
        normalized_question = self._normalize_question(question)
        term = self._extract_follow_up_term(question)
        last_topic = self._last_topic_from_history(chat_history)
        last_condition = self._last_condition_from_history(chat_history)
        topic = term or last_topic or last_condition or "endometriosis atau adenomiosis"
        if (
            not term
            and last_condition
            and ("herbal" in topic or self._is_herbal_intent(topic))
            and last_condition not in topic
        ):
            topic = f"{topic} untuk {last_condition}"
        condition_topic = topic
        if "herbal" in condition_topic or self._is_herbal_intent(condition_topic):
            if "endometriosis" in condition_topic and (
                "adenomiosis" in condition_topic or "adenomyosis" in condition_topic
            ):
                condition_topic = "endometriosis atau adenomiosis"
            elif "endometriosis" in condition_topic:
                condition_topic = "endometriosis"
            elif "adenomiosis" in condition_topic or "adenomyosis" in condition_topic:
                condition_topic = "adenomiosis"
            else:
                condition_topic = "endometriosis atau adenomiosis"

        if any(keyword in normalized_question for keyword in ("gejala", "gejalanya")):
            return f"Apa saja gejala {condition_topic}?"

        if any(
            keyword in normalized_question
            for keyword in ("pemeriksaan", "diagnosis", "diagnosisnya", "didiagnosis")
        ):
            return f"Apa saja pemeriksaan untuk diagnosis {condition_topic}?"

        if any(
            keyword in normalized_question
            for keyword in ("hamil", "kehamilan", "fertilitas", "kesuburan")
        ):
            return (
                f"Bagaimana pilihan terapi {condition_topic} jika pasien ingin hamil "
                "atau mempertahankan kesuburan?"
            )

        if any(
            keyword in normalized_question
            for keyword in (
                "efek samping", "efek sampingnya", "aman", "keamanan",
                "keamanannya", "interaksi", "interaksinya", "dosis", "dosisnya",
                "perhatikan",
            )
        ):
            return (
                f"Apa saja aspek keamanan, efek samping, dosis, dan interaksi "
                f"yang perlu diperhatikan pada {topic}?"
            )

        if any(
            keyword in normalized_question
            for keyword in ("pencegahan", "pencegahannya", "cegah", "mencegah")
        ):
            return f"Bagaimana pencegahan atau pengurangan risiko kekambuhan/gejala pada {condition_topic}?"

        if self._is_herbal_intent(question):
            return (
                f"Bagaimana penggunaan herbal/komplementer untuk {condition_topic} "
                "sebagai pendamping terapi medis?"
            )

        if any(
            keyword in normalized_question
            for keyword in (
                "terapi", "pengobatan", "pengobatannya", "perawatan",
                "penanganan",
            )
        ):
            return f"Bagaimana pilihan terapi atau pengobatan untuk {topic}?"

        if "herbal" in topic or self._is_herbal_intent(topic):
            if any(
                keyword in normalized_question
                for keyword in (
                    "perhatikan", "sebelum", "menggunakannya", "memakainya",
                    "penggunaannya", "pemakaiannya", "aman", "keamanan",
                    "efek samping", "interaksi", "dosis",
                )
            ):
                return (
                    "Apa yang perlu diperhatikan sebelum menggunakan herbal "
                    "untuk endometriosis atau adenomiosis?"
                )
            return f"Bagaimana penggunaan {topic} dalam konteks endometriosis atau adenomiosis?"

        if any(keyword in normalized_question for keyword in ("gejala", "gejalanya")):
            return f"Apa saja gejala {topic}?"

        if any(
            keyword in normalized_question
            for keyword in ("pemeriksaan", "diagnosis", "diagnosisnya", "didiagnosis")
        ):
            return f"Apa saja pemeriksaan untuk diagnosis {topic}?"

        if any(
            keyword in normalized_question
            for keyword in ("hamil", "kehamilan", "fertilitas", "kesuburan")
        ):
            return (
                f"Bagaimana pilihan terapi {topic} jika pasien ingin hamil "
                "atau mempertahankan kesuburan?"
            )

        if any(keyword in normalized_question for keyword in ("terapi", "pengobatan", "pengobatannya")):
            return f"Bagaimana pilihan terapi atau pengobatan untuk {topic}?"

        if term:
            return f"Apa yang dimaksud dengan {term} dalam konteks endometriosis atau adenomiosis?"

        return f"Dalam konteks {topic}, {question}"

    def _has_explicit_condition_scope(self, question: str) -> bool:
        normalized_question = self._normalize_question(question)
        return self._contains_any_keyword(
            normalized_question,
            self.CONDITION_SCOPE_KEYWORDS,
        )

    def _is_contextual_follow_up(self, question: str, chat_history=None) -> bool:
        normalized_question = self._normalize_question(question)
        term = self._extract_follow_up_term(question)
        history_text = self._history_text(chat_history)
        if not history_text or not self._is_in_medical_scope(history_text):
            return False

        has_explicit_condition = self._has_explicit_condition_scope(question)

        if term and len(term) >= 3:
            if term in history_text or self._is_in_medical_scope(term):
                return True

        has_reference = self._contains_any_keyword(
            normalized_question,
            self.FOLLOW_UP_REFERENCE_KEYWORDS,
        )
        has_follow_up_intent = self._contains_any_keyword(
            normalized_question,
            self.FOLLOW_UP_INTENT_KEYWORDS,
        )
        has_care_follow_up_intent = (
            self._contains_any_keyword(normalized_question, self.CARE_SCOPE_KEYWORDS)
            or self._is_herbal_intent(question)
        )
        has_prior_condition = bool(self._last_condition_from_history(chat_history))

        if has_reference:
            return len(normalized_question.split()) <= 14

        if has_follow_up_intent and not has_explicit_condition:
            return len(normalized_question.split()) <= 10

        if has_care_follow_up_intent and has_prior_condition and not has_explicit_condition:
            return len(normalized_question.split()) <= 10

        return False

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
        has_specific_herbal = self._specific_herbal_topic(question) != ""
        has_gynecology_context = self._contains_any_keyword(
            normalized_question,
            self.GYNECOLOGY_CONTEXT_KEYWORDS,
        )
        has_care_context = self._contains_any_keyword(
            normalized_question,
            self.CARE_SCOPE_KEYWORDS,
        )

        return (
            has_condition
            or (has_gynecology_context and has_care_context)
            or (has_specific_herbal and has_care_context)
        )

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
                retrieved_docs = self._retrieve_relevant_docs(contextual_question)
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

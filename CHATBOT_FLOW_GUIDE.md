# 🩺 Rangkuman Aplikasi Chatbot Adenomyosis
## Panduan Lengkap Fitur & Cara Kerja

---

## 📋 DAFTAR ISI
1. [Pengenalan Aplikasi](#pengenalan-aplikasi)
2. [Fitur Utama](#fitur-utama)
3. [Cara Kerja Chatbot](#cara-kerja-chatbot-rag)
4. [Stack Teknologi](#stack-teknologi)
5. [Alur Proses Langkah Demi Langkah](#alur-proses-langkah-demi-langkah)
6. [Komponen-Komponen Utama](#komponen-komponen-utama)
7. [Panduan Penggunaan](#panduan-penggunaan)

---

## Pengenalan Aplikasi

### 🎯 Apa Itu Aplikasi Ini?

**Asisten Ahli: Adenomyosis & Endometriosis** adalah aplikasi web interaktif yang dirancang untuk memberikan informasi medis terpercaya tentang adenomyosis dan endometriosis. Aplikasi menggunakan teknologi **RAG (Retrieval-Augmented Generation)** untuk memberikan jawaban yang didukung oleh literatur ilmiah.

### 📌 Tujuan Utama
- ✅ Edukasi pasien tentang adenomyosis dan endometriosis
- ✅ Menyediakan informasi berbasis bukti ilmiah
- ✅ Visualisasi data medis yang mudah dipahami
- ✅ Chatbot interaktif untuk menjawab pertanyaan pasien
- ✅ Akses mudah ke referensi sumber berkualitas tinggi

### ⚠️ Disclaimer Penting
**Aplikasi ini BUKAN pengganti konsultasi dokter spesialis.** Semua informasi harus dikonfirmasi dengan dokter kandungan (Obgyn) sebelum mengambil keputusan medis.

---

## Fitur Utama

### 1️⃣ **Chat Konsultasi (Chatbot AI)**
Pengguna dapat mengajukan pertanyaan tentang adenomyosis dan endometriosis

**Fitur:**
- Respons instan menggunakan AI Model Gemma 2 / Mistral
- Jawaban berbasis dokumen medis (Metadata Filtering)
- Menampilkan sumber referensi dengan validitas
- Riwayat percakapan yang disimpan
- Mode debug untuk troubleshooting

**Contoh Pertanyaan:**
- "Apa itu adenomyosis?"
- "Apa bedanya adenomyosis dengan endometriosis?"
- "Bagaimana cara mendiagnosis adenomyosis?"
- "Apa pilihan pengobatan terbaik?"

### 2️⃣ **Visualisasi Data (📊 Data Analytics)**
Menampilkan grafik dan diagram interaktif tentang data medis

**Terdapat 5 Jenis Visualisasi:**

| Visualisasi | Deskripsi |
|-------------|-----------|
| 🫀 Anatomi | Diagram struktur rahim dan lokasi adenomyosis |
| 🤕 Gejala | Chart gejala umum adenomyosis |
| 💊 Pengobatan | Opsi pengobatan & efektivitasnya |
| 👥 Distribusi Umur | Grafik adenomyosis per kelompok usia |
| 🔄 Perbandingan | Adenomyosis vs Endometriosis |

### 3️⃣ **FAQ (Frequently Asked Questions)**
10+ pertanyaan umum dengan jawaban lengkap

**Kategori FAQ:**
- ✅ Definisi & Pengenalan
- ✅ Gejala & Diagnosis
- ✅ Pengobatan & Prognosis
- ✅ Kesuburan & Kehamilan
- ✅ Faktor Risiko & Komplikasi

### 4️⃣ **Database Visualizer (Opsional)**
Tools untuk menganalisis database ChromaDB

**Fitur:**
- **Basic Visualizer** - Analisis sederhana database
  - Collection overview
  - Document browsing
  - Metadata exploration
  - Basic statistics

- **Advanced Visualizer** - Analisis mendalam dengan Plotly
  - Network graph (hubungan antar koleksi)
  - Timeline (dokumen per tahun publikasi)
  - Correlation heatmap
  - Content analysis (word frequency)
  - Data export (JSON/CSV)

---

## Cara Kerja Chatbot (RAG)

### 🔄 Apa Itu RAG (Retrieval-Augmented Generation)?

RAG adalah teknologi AI yang menggabungkan:
1. **Retrieval** (Pengambilan): Mencari dokumen relevan dari database
2. **Augmented** (Diperkaya): Menggabungkan dokumen dengan pertanyaan
3. **Generation** (Pembuatan): Model AI menghasilkan jawaban berbasis dokumen

```
Pertanyaan User
      ↓
[Retriever mencari dokumen serupa di ChromaDB]
      ↓
[Dokumen relevan dikumpulkan + disambung]
      ↓
[LLM (Language Model) membaca konteks + pertanyaan]
      ↓
[Model menghasilkan jawaban berbasis dokumen]
      ↓
Jawaban + Sumber Referensi
```

### 🎯 Keuntungan RAG untuk Chatbot Medis

| Aspek | Keuntungan |
|-------|-----------|
| **Akurasi** | Jawaban didukung dokumen ilmiah, bukan "halusinasi" AI |
| **Transparansi** | Menampilkan sumber rujukan yang dapat diverifikasi |
| **Relevansi** | Fokus pada topik adenomyosis/endometriosis |
| **Terpercaya** | Metadata filtering memastikan sumber berkualitas |
| **Local** | Tidak perlu cloud API internet-intensive |

---

## Alur Proses Langkah Demi Langkah

### 📍 FASE 1: INISIALISASI APLIKASI

```
1. Pengguna buka app.py dengan Streamlit
   └─ streamlit run app.py

2. ChatBot class diinisialisasi (@st.cache_resource)
   ├─ Load HuggingFace API Token dari st.secrets
   ├─ Setup Embedding Model (paraphrase-multilingual)
   └─ Setup ChromaDB dengan persist_directory

3. ChromaDB Checking:
   ├─ Jika database EXISTS
   │  └─ Load database dari chroma_db_adenomyosis/
   └─ Jika TIDAK EXISTS
      ├─ Load semua PDF dari folder data_adenomyosis/
      ├─ Extract text dengan PyMuPDF (fitz)
      ├─ Split dokumen dengan RecursiveCharacterTextSplitter
      ├─ Create embeddings untuk setiap chunk
      └─ Save ke ChromaDB (persistent storage)

4. RAG Chain setup
   ├─ Inisialisasi Retriever
   ├─ Setup LLM client (HuggingFace Serverless)
   └─ Compile RAG pipeline
```

### 📍 FASE 2: USER INTERFACE RENDER

```
1. Render Header
   ├─ Judul: "🩺 Asisten Ahli: Adenomyosis & Endometriosis"
   ├─ Penjelasan teknologi RAG + ChromaDB
   └─ Author credit

2. Render Sidebar Navigation
   ├─ Radio button: Pilih halaman
   │  ├─ 💬 Chat Konsultasi
   │  ├─ 📊 Visualisasi Data
   │  └─ 📚 Informasi Umum
   ├─ Debug Mode toggle
   └─ Disclaimer medis
```

### 📍 FASE 3: USER INPUT PERTANYAAN

```
1. User ketik pertanyaan di chat input
   └─ Contoh: "Apa itu adenomyosis?"

2. Pertanyaan ditambahkan ke st.session_state.messages

3. System menampilkan indicator "Menganalisis dokumen medis..."
   └─ (Spinner animation)
```

### 📍 FASE 4: RETRIEVAL (PENGAMBILAN DOKUMEN)

```
1. Pertanyaan dikonversi ke vector embedding
   ├─ Model: sentence-transformers/paraphrase-multilingual
   └─ Dimensi: 384-dimensional vector

2. Semantic search di ChromaDB
   ├─ Cari 5 dokumen paling mirip (k=5)
   ├─ Menggunakan similarity search
   └─ Return: Top-5 dokumen + metadata

3. Dokumen yang diambil diformat dengan metadata:
   ├─ [Sumber: Jurnal Ilmiah | Validitas: High | Tahun: 2023]
   ├─ Konten dokumen chunk
   └─ ...repeat untuk 5 dokumen
```

### 📍 FASE 5: LLM INFERENCE (PEMBUATAN JAWABAN)

```
1. Prepare messages untuk LLM:
   ├─ System Message: "Anda adalah asisten medis ahli..."
   └─ User Message:
      ├─ Konteks (5 dokumen retrieved)
      ├─ Pertanyaan pasien
      └─ Instruksi format (2-3 paragraf + saran konsultasi)

2. Kirim ke HuggingFace Serverless Inference
   ├─ Model: Mixtral-8x7B-Instruct (primary)
   │  Fallback: Meta-Llama-3-8B, Microsoft Phi-3
   ├─ Max tokens: 400
   ├─ Temperature: 0.7 (creative tapi focused)
   └─ Top_p: 0.9

3. Receive response dari LLM
   ├─ Extract jawaban dari response.choices[0].message.content
   └─ Error handling dengan fallback response
```

### 📍 FASE 6: RESPONSE FORMATTING

```
1. Format jawaban:
   ├─ Paragraf 1-2: Jawaban utama (evidence-based)
   ├─ Paragraf 3: Rekomendasi "Konsultasi dokter spesialis"
   └─ Max: 2-3 paragraf (readability)

2. Extract sumber referensi dari retrieved docs:
   ├─ Source filename
   ├─ Metadata:
   │  ├─ validity_level (high/medium/low)
   │  ├─ source_type (Jurnal/Guideline/Buku/Review)
   │  └─ year (tahun publikasi)
   └─ Badge display: 🟢High / 🟡Medium / ⚪Low
```

### 📍 FASE 7: DISPLAY HASIL

```
1. Render jawaban chatbot
   ├─ Role: "assistant"
   ├─ Content: Jawaban full text
   └─ Markdown formatting: **bold**, ✅ bullets, dsb

2. Render Sumber Referensi
   ├─ Judul: "📚 Sumber Referensi:"
   ├─ Untuk setiap sumber:
   │  └─ • **Filename** | Type | Badge | Tahun
   └─ Contoh:
      └─ • **adenomyosis_comprehensive_review** | Review Article | 🟢High | 2023

3. Simpan ke session history
   ├─ messages.append({
   │  ├─ role: "assistant"
   │  ├─ content: jawaban
   │  ├─ sources: [list file referensi]
   │  └─ metadata: {source: {validity, type, year}}
   └─ })
```

### 📍 FASE 8: DISPLAY HISTORY

```
Pertemuan berikutnya/pertanyaan baru:

1. Semua pesan sebelumnya ditampilkan:
   ├─ User message (role: "user")
   └─ Assistant message (role: "assistant")
      ├─ Jawaban teks
      ├─ Divider (---)
      └─ Sumber referensi dengan metadata

2. Session state disimpan selama sesi browser
   └─ Jika refresh/close browser → history hilang (gunakan st.session_state)
```

---

## Stack Teknologi

### 🔧 Backend Framework & Libraries

| Komponen | Library | Fungsi |
|----------|---------|--------|
| **Frontend** | Streamlit | UI web interaktif |
| **Vector DB** | ChromaDB | Penyimpanan embedding vektor |
| **Embedding** | Sentence-Transformers | Konversi teks → vektor |
| **PDF Processing** | PyMuPDF (fitz) | Extract teks dari PDF |
| **Text Splitting** | RecursiveCharacterTextSplitter | Chunking dokumen |
| **LLM Framework** | LangChain | Orchestration RAG pipeline |
| **LLM Inference** | HuggingFace Hub | Serverless inference |
| **Visualization** | Plotly, Altair | Interactive charts |

### 🤖 AI Models

| Model | Fungsi | Provider |
|-------|--------|----------|
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Embedding teks (multilingual) | HuggingFace |
| `mistralai/Mixtral-8x7B-Instruct-v0.1` | LLM utama untuk jawaban | HuggingFace Serverless |
| `meta-llama/Meta-Llama-3-8B-Instruct` | LLM fallback #1 | HuggingFace Serverless |
| `microsoft/Phi-3-mini-4k-instruct` | LLM fallback #2 (lightweight) | HuggingFace Serverless |

### 🗄️ Database Structure

```
chroma_db_adenomyosis/
├── chroma.sqlite3          # Main database
├── data/                   # Vector data storage
└── index/                  # Index for fast retrieval

ChromaDB Collection Schema:
├─ collection_name: "adenomyosis_docs"
├─ embedding_dimension: 384
├─ distance_metric: cosine
├─ documents: [text chunks]
├─ embeddings: [384-dim vectors]
└─ metadata per doc:
   ├─ source (filename)
   ├─ source_type (Jurnal/Guideline/Buku)
   ├─ validity_level (high/medium/low)
   ├─ year (publication year)
   ├─ category (adenomyosis/endometriosis/diagnosis/treatment)
   ├─ page_count
   └─ chunk_id
```

### 📦 Dependencies

```
streamlit               # Web framework
pillow                  # Image processing
pandas                  # Data manipulation
altair                  # Charts (basic)
plotly                  # Charts (advanced)
networkx                # Network visualization
langchain==0.2.16       # RAG orchestration
langchain-community     # Additional components
langchain-core          # Core interfaces
langchain-huggingface   # HF integration
langchain-chroma        # ChromaDB integration
chromadb==0.4.22        # Vector database
huggingface-hub         # HF API client
transformers            # Model utilities
sentence-transformers   # Embedding models
pymupdf                 # PDF extraction
python-dotenv           # Environment variables
```

---

## Komponen-Komponen Utama

### 1. **app.py** - Main Application Entry Point
```
Fungsi Utama:
├─ get_chatbot()           → Initialize ChatBot (cached)
├─ render_header()         → Tampilkan judul & intro
├─ render_sidebar()        → Tampilkan navigasi
├─ render_chat_page()      → Halaman chat interface
├─ render_visualization_page() → Halaman grafik
├─ render_info_page()      → Halaman FAQ
└─ main()                  → Main execution loop

Alur Eksekusi:
1. Config Streamlit page (icon, layout)
2. Load environment variables (.env)
3. Initialize ChatBot (one-time)
4. Render header & sidebar
5. Conditional render berdasarkan halaman yang dipilih
6. Handle user input (st.chat_input)
7. Call bot.ask(question)
8. Format & display response + sources
```

### 2. **chatbot.py** - RAG Pipeline & LLM Logic
```
Class: ChatBot

Constructor Methods:
├─ __init__()
├─ _resolve_persist_directory()      → Find writable DB location
├─ _initialize_hf_client()           → Setup HF API + models
├─ _initialize_chroma()              → Load/create ChromaDB
└─ _setup_rag_chain()                → Build RAG pipeline

PDF Loading Methods:
├─ _load_pdfs_from_folder()          → Load all PDFs
├─ _extract_metadata_from_filename() → Parse file for metadata
└─ [Private] Metadata extraction:
   ├─ source_type detection (Jurnal/Guideline/Buku/Review)
   ├─ validity_level assignment (high/medium/low)
   ├─ year extraction from filename
   └─ category classification

RAG Pipeline Methods:
├─ ask(question)                     → Main query method
├─ [Format docs function]            → Dengan metadata badge
├─ [LLM call function]               → Serverless inference
├─ [Error handling]                  → Fallback responses
└─ _generate_fallback_response()     → Generic response saat error

Key Features:
✅ Metadata filtering (source validation)
✅ Semantic chunking (context preservation)
✅ Multi-model fallback strategy
✅ Comprehensive error handling
✅ ChromaDB persistence
```

### 3. **common_questions.py** - FAQ Database
```
COMMON_QUESTIONS Dictionary:

Structure:
{
  "Question text": {
    "question": "Full question",
    "answer": "Detailed answer (2-3 paragraphs)",
    "category": "Definisi/Gejala/Diagnosis/Pengobatan/Kehamilan/etc"
  }
}

10 Pertanyaan Coverage:
1. Definisi adenomyosis
2. Perbedaan adenomyosis & endometriosis
3. Gejala
4. Diagnosis
5. Pengobatan
6. Infertilitas
7. Prognosis
8. Keamanan (non-kanker)
9. Pengaruh pada kehamilan
10. Faktor risiko

QUICK_TOPICS: List tombol topic cepat
```

### 4. **utils.py** - Utility Functions
```
Main Functions:
├─ initialize_session_state()
│  ├─ st.session_state.messages (chat history)
│  ├─ st.session_state.current_page (navigation)
│  └─ st.session_state.debug_mode (debugging toggle)
└─ [Additional utilities if any]
```

### 5. **visualizations.py** - Chart Generation
```
Visualization Functions:
├─ show_adenomyosis_diagram()        → Anatomical diagram
├─ show_symptoms_chart()             → Symptoms comparison
├─ show_treatment_options()          → Treatment effectiveness
├─ show_age_distribution()           → Age groups chart
└─ show_adenomyosis_vs_endometriosis() → Side-by-side comparison

Implementation: Streamlit components (st.bar_chart, st.plotly_chart, etc)
```

### 6. **Database Visualizers** (Optional)

#### db_visualizer.py - Basic Analysis
```
4 Pages:
1. Database Overview
   ├─ Total collections & documents
   ├─ Metrics dashboard
   └─ Collection statistics

2. Collections & Documents
   ├─ Browse documents
   ├─ View metadata
   └─ Sample data table

3. Schema & Model
   ├─ Entity relationship diagram
   └─ Collection structure

4. Statistics
   ├─ Document count per collection
   ├─ Charts (Altair)
   └─ Metadata analysis
```

#### db_visualizer_advanced.py - Advanced Analysis
```
5 Interactive Tabs:
1. Network Graph
   ├─ Collection relationship network
   └─ Interactive node visualization

2. Timeline
   ├─ Documents over years
   ├─ Publication trends
   └─ Plotly timeline chart

3. Metadata Correlation
   ├─ Co-occurrence analysis
   └─ Heatmap visualization

4. Size Distribution
   ├─ Documents per collection
   └─ Pie chart

5. Content Analysis
   ├─ Word frequency analysis
   ├─ Metadata insights
   └─ Export capabilities
```

---

## Panduan Penggunaan

### ✅ Setup Awal (First Time)

#### Langkah 1: Install Dependencies
```bash
# Navigasi ke folder project
cd "e:\NLP\2026 NLP\Chatbot Adenom Baru\chatbot_endo_adenom"

# Install semua packages
pip install -r requirements.txt
```

#### Langkah 2: Setup Environment Variables
Buat file `.env` di root folder:
```env
HUGGINGFACE_API_KEY=hf_xxxxxxxxxxxxxxxxxxxxx
STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
```

Dapatkan HuggingFace API Key dari: https://huggingface.co/settings/tokens

#### Langkah 3: Prepare Data
Pastikan folder `data_adenomyosis/` berisi PDF files dengan nama deskriptif:
```
data_adenomyosis/
├─ adenomyosis_comprehensive_review_2023.pdf
├─ clinical_guideline_adenomyosis_2022.pdf
├─ adenomyosis_treatment_options_2024.pdf
├─ endometriosis_pathophysiology_2021.pdf
└─ diagnosis_imaging_adenomyosis_2023.pdf
```

**Naming Convention untuk optimal metadata extraction:**
- Gunakan tahun publikasi: `2023`, `2022`, dst
- Sertakan tipe dokumen:
  - `journal` / `jurnal` → Jurnal Ilmiah (validitas: high)
  - `guideline` / `pedoman` / `clinical` → Guideline Klinis (validitas: very_high)
  - `textbook` / `buku` / `book` → Buku Teks (validitas: high)
  - `review` / `tinjauan` → Review Article (validitas: high)
- Sertakan kategori:
  - `adenomyosis` / `adenom` → adenomyosis
  - `endometriosis` / `endo` → endometriosis
  - `diagnosis` / `diagnosa` → diagnosis
  - `treatment` / `pengobatan` / `terapi` → treatment

Contoh nama file baik:
```
adenomyosis_clinical_guideline_diagnosis_2023.pdf
endometriosis_journal_treatment_options_2022.pdf
adenomyosis_comprehensive_review_pathophysiology_2024.pdf
```

#### Langkah 4: Run Aplikasi
```bash
# Terminal: Activate virtual environment (already done based on context)

# Run aplikasi
streamlit run app.py
```

Browser akan otomatis membuka `http://localhost:8501`

### 💬 Menggunakan Chat

#### Cara Bertanya
1. Buka halaman **"💬 Chat Konsultasi"** dari sidebar
2. Ketik pertanyaan di input box
3. Tekan Enter atau klik Send
4. Tunggu AI menganalisis (2-5 detik)
5. Baca jawaban + sumber referensi

#### Tips Pertanyaan Efektif
✅ **Baik:**
- "Apa perbedaan adenomyosis dengan endometriosis?"
- "Apa saja gejala adenomyosis yang harus saya ketahui?"
- "Bagaimana diagnosis adenomyosis dilakukan?"
- "Apa pilihan pengobatan terbaik untuk adenomyosis?"

❌ **Kurang Baik:**
- "Adenomyosis?" (terlalu singkat)
- "Apa-apa saja tentang penyakit wanita?" (terlalu umum)
- Pertanyaan bukan tentang adenomyosis/endometriosis

#### Memahami Sumber Referensi
```
Contoh ditampilkan:
• **adenomyosis_clinical_guideline_2023** | Guideline Klinis | 🟢 High | 2023

Badge Validitas:
🟢 High    → Guideline klinis resmi / Jurnal terpercaya
🟡 Medium  → Jurnal umum / Buku teks
⚪ Low     → Sumber umum / Data tidak terverifikasi
```

### 📊 Visualisasi Data

1. Buka halaman **"📊 Visualisasi Data"** dari sidebar
2. Pilih tab visualisasi yang diinginkan:
   - 🫀 **Anatomi** - Bagian struktur rahim
   - 🤕 **Gejala** - Gejala umum adenomyosis
   - 💊 **Pengobatan** - Opsi pengobatan
   - 👥 **Distribusi Umur** - Prevalansi berdasarkan usia
   - 🔄 **Perbandingan** - Adenomyosis vs Endometriosis
3. Hover di chart untuk melihat detail
4. Klik legend untuk filter (jika supported)

### 📚 FAQ

1. Buka halaman **"📚 Informasi Umum"** dari sidebar
2. Klik pada pertanyaan yang ingin dibuka (expandable)
3. Baca jawaban lengkap

### 🐛 Debug Mode

Untuk troubleshooting:
1. Di sidebar, toggle **"🐛 Debug Mode"**
2. Saat chatbot error, akan menampilkan:
   - Error type (exception class)
   - Error message
   - Model yang digunakan
   - Token verification status
3. Gunakan untuk diagnose masalah koneksi HF, token, dll

---

## 📈 Performance & Optimization Tips

### Kecepatan Respon
- **First query**: 3-8 detik (model loading)
- **Subsequent queries**: 2-5 detik (model cached)
- **RAG retrieval**: < 1 detik
- **LLM inference**: 1-4 detik (tergantung model)

### Tips Optimasi
1. **Use cache untuk ChatBot instance**
   - `@st.cache_resource def get_chatbot()`
   - Mencegah reinisialisasi setiap interaction

2. **Metadata filtering**
   - Filter sumber berdasarkan validity_level
   - Prioritas: very_high → high → medium

3. **Semantic chunking**
   - chunk_size: 500 characters (balanced)
   - chunk_overlap: 150 characters (context preservation)

4. **Database persistence**
   - ChromaDB cache embeddings locally
   - Tidak perlu re-embed setiap run

### Troubleshooting Umum

| Problem | Solution |
|---------|----------|
| "No module named 'streamlit'" | `pip install -r requirements.txt` |
| "HUGGINGFACE_API_KEY not found" | Setup `.env` file dengan token |
| "ChromaDB connection error" | Delete `chroma_db_adenomyosis/` & restart |
| "Model loading failed" | Check HF token validity, fallback model akan digunakan |
| "Empty response from LLM" | Pertanyaan mungkin di luar scope, coba rephrase |
| "Slow response time" | HF models might be loading, tunggu 1-2 menit |

---

## 🎓 Arsitektur Sistem (High Level)

```
┌─────────────────────────────────────────────────────────────┐
│                    STREAMLIT WEB UI                         │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Sidebar: Navigation + Debug Toggle                  │  │
│  │ Header: Title + Technology Explanation               │  │
│  │ Main Content Area:                                    │  │
│  │ ├─ Chat Interface (Chat Konsultasi)                 │  │
│  │ ├─ Visualizations (Visualisasi Data)                │  │
│  │ └─ FAQ (Informasi Umum)                             │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      RAG PIPELINE (LangChain)               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 1. User Question Input                              │  │
│  │ 2. Retriever: Semantic search (ChromaDB)            │  │
│  │    └─ Top-5 similar documents                       │  │
│  │ 3. Format Docs: Add metadata, context               │  │
│  │ 4. LLM Call: Send to HuggingFace                    │  │
│  │ 5. Format Response: Extract answer                  │  │
│  │ 6. Return: Answer + Sources                         │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      EXTERNAL SERVICES                      │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ HuggingFace Hub (Serverless Inference):             │  │
│  │ ├─ Mixtral-8x7B-Instruct (primary)                  │  │
│  │ ├─ Meta-Llama-3-8B-Instruct (fallback 1)            │  │
│  │ └─ Microsoft Phi-3-mini (fallback 2)                │  │
│  │                                                      │  │
│  │ Embedding Model:                                     │  │
│  │ └─ sentence-transformers/paraphrase-multilingual   │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                     LOCAL DATA LAYER                        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ ChromaDB (Vector Database):                         │  │
│  │ ├─ chroma_db_adenomyosis/chroma.sqlite3             │  │
│  │ ├─ Vector embeddings (384-dim)                      │  │
│  │ └─ Metadata (source, year, validity, category)      │  │
│  │                                                      │  │
│  │ Data Files:                                          │  │
│  │ ├─ data_adenomyosis/*.pdf (source documents)        │  │
│  │ └─ config.toml (configuration)                      │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 📌 Key Takeaways

### ✅ Apa yang Aplikasi Ini Lakukan Dengan Baik
1. **Evidence-Based Answers** - Jawaban didukung dokumen ilmiah
2. **Source Transparency** - Setiap jawaban menampilkan referensi
3. **Metadata Filtering** - Hanya sumber berkualitas tinggi
4. **Multilingual Support** - Embedding model mendukung Bahasa Indonesia + 100+ bahasa
5. **Low Latency** - Retrieval + generation < 5 detik
6. **Local-First** - Semua data disimpan lokal, privacy-preserving

### ⚠️ Limitations
1. **Scope Terbatas** - Hanya untuk adenomyosis/endometriosis
2. **Knowledge Cutoff** - Hanya dokumen dalam folder data_adenomyosis
3. **Not Real Medical Advice** - Selalu konsultasi dokter
4. **No User Authentication** - Semua user akses data yang sama
5. **Single-Session Memory** - Chat history hilang jika refresh browser

### 🚀 Potential Improvements
1. **Persistent Chat History** - Save ke database
2. **User Authentication** - Per-user chat history
3. **Document Search UI** - Search documents directly
4. **Feedback System** - Rate response quality
5. **Multi-language UI** - English, Arabic, etc
6. **Export Reports** - Generate PDF reports dari chat
7. **Streaming Responses** - Real-time answer streaming
8. **Citation System** - Clickable citations dengan page numbers
9. **Doctor Dashboard** - Admin panel untuk manage documents
10. **Mobile App** - React Native / Flutter version

---

## 📞 Support & Contact

**Aplikasi Ini Dibuat Oleh:**
- **Nuraisa Novia Hidayati** (Riset Prototipe)

**Untuk Pertanyaan Teknis:**
- Periksa DEBUG MODE untuk error details
- Lihat terminal output untuk logs
- Verify `.env` file setup

**Medical Disclaimer:**
- **BUKAN pengganti konsultasi dokter**
- Semua keputusan medis harus dikonfirmasi dengan Obgyn
- Informasi ini untuk edukasi pasien saja

---

## 📎 Appendix: Quick Reference

### File Structure
```
chatbot_endo_adenom/
├── app.py                          # Main app
├── chatbot.py                      # RAG logic
├── common_questions.py             # FAQ database
├── utils.py                        # Utilities
├── visualizations.py               # Charts
├── evaluation.py                   # Evaluation tools
├── config.toml                     # Configuration
├── requirements.txt                # Dependencies
├── .env                            # Environment (not in repo)
├── chroma_db_adenomyosis/          # Vector DB (auto-created)
│   └── chroma.sqlite3
├── data_adenomyosis/               # Source PDFs (user provides)
│   └── *.pdf
├── FEATURES_SUMMARY.md             # Database visualizer features
├── QUICKSTART_GUIDE.md             # Quick setup guide
├── DATABASE_VISUALIZER_README.md   # Detailed visualizer doc
├── INTEGRATION_GUIDE.md            # Integration info
└── CHATBOT_FLOW_GUIDE.md          # This file!
```

### Environment Variables
```env
HUGGINGFACE_API_KEY=hf_...    # Required
STREAMLIT_SERVER_FILE_WATCHER_TYPE=none   # Recommended
ANONYMIZED_TELEMETRY=False    # For ChromaDB
POSTHOG_DISABLED=1             # For ChromaDB
```

### LLM Model Parameters
```python
max_tokens: 400           # Response length limit
temperature: 0.7          # Creativity level (0=deterministic, 1=random)
top_p: 0.9               # Nucleus sampling parameter
search_type: "similarity"  # ChromaDB retrieval method
k: 5                      # Number of documents retrieved
```

---

**Versi Dokumen:** 1.0  
**Terakhir Diupdate:** May 25, 2026  
**Status:** Production

# 📊 Database Visualizer - Dokumentasi

## Pengenalan

Database Visualizer adalah aplikasi terpisah untuk menganalisis dan memvisualisasikan struktur database ChromaDB yang digunakan oleh Adenomyosis Chatbot.

## Fitur Utama

### 1. **Database Overview** 📊
- Menampilkan total collections dan documents
- Metrics untuk setiap collection
- Sample data dari documents
- Metadata overview

### 2. **Collections & Documents Explorer** 📁
- Browse semua documents dalam collection
- Tampilkan document preview dengan metadata
- Visualisasi distribusi metadata dengan chart interaktif
- Filter dan analisis metadata

### 3. **Schema & Data Model** 🗂️
- Entity-Relationship Diagram (ERD)
- Data model untuk Adenomyosis application
- Document types dan metadata fields explanation
- Data flow visualization

### 4. **Statistics** 📈
- Statistik collections
- Documents per collection
- Metadata distribution analysis
- Performance metrics

## Cara Menjalankan

### Method 1: Run Visualizer Standalone
```bash
# Navigasi ke folder aplikasi
cd "e:\NLP\2026 NLP\Chatbot Adenom Baru\chatbot_endo_adenom"

# Run visualizer sebagai standalone app
streamlit run db_visualizer.py
```

Aplikasi akan membuka di `http://localhost:8501`

### Method 2: Integrasi ke Main App
Untuk menambahkan visualizer ke main app (app.py), dapat membuat multipage Streamlit app:

```bash
# Struktur direktori recommended:
chatbot_endo_adenom/
├── app.py                    # Main application
├── db_visualizer.py          # Standalone visualizer
├── pages/
│   └── 📊_Database_Viz.py   # Database visualizer page
├── chatbot.py
├── visualizations.py
└── utils.py
```

## Interface Walkthrough

### Dashboard Overview
```
┌─────────────────────────────────────────┐
│ Database Visualization & Analytics      │
├─────────────────────────────────────────┤
│ Total Collections: X    Total Docs: Y   │
├─────────────────────────────────────────┤
│ Collections Expandable List:            │
│  ├─ Collection Name (Count documents)   │
│  │  ├─ Documents table                  │
│  │  ├─ Metadata analysis                │
│  │  └─ Sample documents                 │
│  └─ ...                                 │
└─────────────────────────────────────────┘
```

### Navigation Sidebar
- **Database Overview**: Main dashboard dengan collection metrics
- **Collections & Documents**: Explore documents dan metadata
- **Schema & Model**: Lihat database structure dan data model
- **Statistics**: Analytics dan charts

## Database Structure

### ChromaDB Collections Hierarchy
```
ChromaDB
└── Collections
    └── [Collection Name]
        ├── Documents
        │   ├── id (UUID)
        │   ├── content (text)
        │   ├── embedding (vector)
        │   └── metadata (key-value)
        ├── Metadata Schema
        │   ├── source (filename)
        │   ├── page (number)
        │   ├── category (type)
        │   ├── year (publication)
        │   └── chunk_id (identifier)
        └── Vector Index (similarity search)
```

## Metadata Fields Explained

| Field | Type | Example | Purpose |
|-------|------|---------|---------|
| `source` | string | "jurnal_adenomiosis_2023.pdf" | Document file source |
| `page` | number | 5 | Page number in PDF |
| `category` | string | "Research Paper" | Document classification |
| `year` | number | 2023 | Publication year |
| `chunk_id` | string | "chunk_001" | Text segment identifier |
| `relevance` | number | 0.87 | Similarity score |

## Document Processing Pipeline

```
PDF Files (data_adenomyosis/)
        ↓
    [1] Text Extraction
        PyMuPDF (fitz) reads PDF content
        ↓
    [2] Text Chunking
        RecursiveCharacterTextSplitter
        - Overlap for context preservation
        - Metadata attached to chunks
        ↓
    [3] Embedding Generation
        sentence-transformers/paraphrase-multilingual
        - Semantic understanding of medical terms
        - Supports Indonesian language
        ↓
    [4] ChromaDB Storage
        - Vector index for similarity search
        - Metadata filtering
        - Efficient retrieval
        ↓
    [5] Query Processing (at runtime)
        - User query embedding
        - Similarity search (top-k)
        - Metadata filtering
        - Results ranking
        ↓
    [6] LLM Generation
        - Context from retrieved documents
        - Generate answer with sources
        - Turkish language support
```

## Common Queries

### Query 1: View All Documents
```python
client = get_chroma_client()
collection = client.get_collection(name="default")
all_docs = collection.get(limit=1000)
```

### Query 2: Filter by Metadata
```python
results = collection.query(
    query_texts=["adenomyosis treatment"],
    where={"year": {"$gte": 2020}},
    n_results=10
)
```

### Query 3: Get Collection Stats
```python
doc_count = collection.count()
print(f"Total documents: {doc_count}")
```

## Troubleshooting

### Issue: "ChromaDB Connection Failed"
**Solution**: 
- Pastikan folder `chroma_db_adenomyosis/` exists dan writable
- Check disk space
- Restart Streamlit

### Issue: "No Collections Found"
**Solution**:
- Jalankan main chatbot app terlebih dahulu untuk load documents
- ChromaDB akan auto-create collections saat pertama kali

### Issue: "Slow Query Performance"
**Solution**:
- Kurangi limit pada `collection.get(limit=...)`
- Gunakan metadata filtering untuk reduce scope
- Check available memory

## Advanced Features

### Custom Visualizations
Edit `db_visualizer.py` untuk menambah:
- Network graph visualization
- Temporal analysis (documents over time) 
- Geographic distribution
- Topic clustering and heatmaps

### Export Data
```python
# Export ke CSV
df.to_csv('collections_export.csv', index=False)

# Export to JSON
with open('db_snapshot.json', 'w') as f:
    json.dump(analysis, f, indent=2)
```

### Performance Monitoring
```python
# Track query performance
import time

start = time.time()
results = collection.query(...)
duration = time.time() - start
print(f"Query time: {duration:.2f}s")
```

## Integration dengan Main App

Untuk menambahkan ke navbar utama:

```python
# Di app.py sidebar
with st.sidebar:
    page = st.radio("Navigation", [
        "💬 Chat",
        "📊 Visualisasi",
        "📚 Info",
        "🗄️ Database Viz"  # <-- Add this
    ])
    
    if page == "🗄️ Database Viz":
        import db_visualizer
        db_visualizer.main()
```

## Kontribusi

Untuk menambah fitur visualization:
1. Edit file `db_visualizer.py`
2. Tambah function baru sesuai pattern existing
3. Add ke navigation di `main()`
4. Test dengan sample data

## Lihat Juga

- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Streamlit API Reference](https://docs.streamlit.io/)
- [Altair Visualization Gallery](https://altair-viz.github.io/)

---

**Last Updated**: April 2026  
**Version**: 1.0  
**Author**: Database Visualization System

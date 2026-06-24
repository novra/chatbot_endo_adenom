# 🚀 Database Visualizer - Quick Start Guide

## 📋 Checklist Setup

### ✅ Step 1: Install Dependencies
```bash
# Navigasi ke folder aplikasi
cd "e:\NLP\2026 NLP\Chatbot Adenom Baru\chatbot_endo_adenom"

# Install packages
pip install -r requirements.txt

# Verify install
pip show streamlit chromadb plotly pandas altair
```

### ✅ Step 2: Verify Database Exists
ChromaDB database akan dibuat otomatis saat aplikasi utama pertama kali dijalankan:
```
chroma_db_adenomyosis/
├── data/
├── index/
└── chroma.sqlite3
```

### ✅ Step 3: Run Standalone Visualizer
```bash
streamlit run db_visualizer.py
```

**Output yang diharapkan:**
```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
Network URL: http://[your-ip]:8501
```

### ✅ Step 4: Run Advanced Visualizer
```bash
streamlit run db_visualizer_advanced.py
```

---

## 🎯 Feature Comparison

| Feature | Basic (`db_visualizer.py`) | Advanced (`db_visualizer_advanced.py`) |
|---------|---------------------------|---------------------------------------|
| Collection Overview | ✅ | ✅ |
| Documents Browse | ✅ | ✅ |
| Metadata Analysis | ✅ With Altair | ✅ With Plotly |
| Network Graph | ❌ | ✅ Interactive |
| Timeline Visualization | ❌ | ✅ |
| Correlation Matrix | ❌ | ✅ Heatmap |
| Size Distribution | ❌ | ✅ Pie Chart |
| Content Analysis | ❌ | ✅ Word Frequency |
| Export Data | ❌ | ✅ JSON/Summary |
| Performance | Fast | Detailed |

---

## 📊 Visualizer Interface Guide

### Basic Visualizer (`db_visualizer.py`)

#### Page 1: Database Overview
```
┌──────────────────────────────────────┐
│      DATABASE VISUALIZATION          │
│  Total Collections: X | Total Docs: Y│
│                                      │
│  Collections:                         │
│  ├─ Collection 1 (500 docs)          │
│  │  • 12 metadata keys               │
│  │  • Sample documents table         │
│  ├─ Collection 2 (1,200 docs)        │
│  │  • 15 metadata keys               │
│  │  • Sample documents table         │
└──────────────────────────────────────┘
```

#### Page 2: Collections & Documents
```
┌──────────────────────────────────────┐
│    COLLECTIONS & DOCUMENTS           │
│  Select Collection: [Dropdown ▼]     │
│                                      │
│  📄 Documents Table                  │
│  ID | Preview | Metadata cols...     │
│  ---|---------|--------------------  │
│                                      │
│  📊 Metadata Distribution Chart      │
│  Select Field: [category ▼]          │
│  Bar chart showing distribution      │
└──────────────────────────────────────┘
```

#### Page 3: Schema & Model
- ChromaDB system architecture
- Adenomyosis data model explanation
- Data processing pipeline diagram
- Field descriptions

#### Page 4: Statistics
- Collection metrics table
- Documents per collection chart
- Comparison statistics

### Advanced Visualizer (`db_visualizer_advanced.py`)

#### Tab 1: Network Graph
```
Interactive visualization:
- Central ChromaDB node (red)
- Collection nodes (teal)
- Metadata field nodes (light teal)
- Edges showing relationships
- Hover for details
```

#### Tab 2: Timeline
```
Bar chart showing:
- X-axis: Publication year
- Y-axis: Number of documents
- Color-coded by frequency
```

#### Tab 3: Metadata Correlation
```
Heatmap showing:
- Field co-occurrence matrix
- Blue intensity = correlation strength
- Helps understand metadata patterns
```

#### Tab 4: Size Distribution
```
Pie chart showing:
- Document count per collection
- Percentage distribution
- Total documents info
```

#### Tab 5: Content Analysis
```
Bar chart showing:
- Document type frequency
- Source distribution
- Most common metadata values
```

---

## 🔧 Common Use Cases

### Use Case 1: Check Database Status
```python
# Quick check if database is populated
streamlit run db_visualizer.py

# Navigate: Database Overview
# Check: Total Collections and Documents counts
```

### Use Case 2: Explore Documents
```python
# Browse what documents are in database
streamlit run db_visualizer.py

# Navigate: Collections & Documents
# Select collection → View Table → Search documents
```

### Use Case 3: Analyze Metadata
```python
# Understand metadata distribution
streamlit run db_visualizer_advanced.py

# Tab: Metadata Correlation
# Visualize which fields are related
```

### Use Case 4: Quality Check
```python
# Verify document distribution
streamlit run db_visualizer_advanced.py

# Tab: Timeline → Check year distribution
# Tab: Size Distribution → Verify balanced loading
```

---

## 📈 Interpreting Results

### Document Count Analysis
```
If you see:
- ✅ 100+ documents per collection → Good coverage
- ⚠️  50-100 documents → Limited coverage
- ❌ <50 documents → Possibly incomplete data loading
```

### Metadata Completeness
```
If you see:
- ✅ All documents have 'source' metadata → Good
- ⚠️  50-80% have metadata → Missing some
- ❌ <50% have metadata → Data quality issue
```

### Timeline Distribution
```
If you see:
- ✅ Documents from multiple years (2019-2024) → Diverse sources
- ⚠️  All from 2023 only → Limited temporal scope
- ❌ No year metadata → Cannot verify publication dates
```

---

## 🐛 Troubleshooting

### Error: "ChromaDB Connection Failed"
```
Solution:
1. Check folder exists: chroma_db_adenomyosis/
2. Run main app first: streamlit run app.py
   (This initializes the database)
3. Then run visualizer: streamlit run db_visualizer.py
4. If still fails: Delete chroma_db_adenomyosis/ and restart
```

### Error: "No Collections Found"
```
Solution:
1. Main database hasn't been initialized yet
2. Steps:
   a. Copy your PDF files to data_adenomyosis/
   b. Run main app: streamlit run app.py
   c. Wait for "Loaded documents" message
   d. Then run visualizer
```

### Error: "Slow Loading / Timeout"
```
Solution:
1. Reduce document limit in code
   - Change: collection.get(limit=5000)
   - To: collection.get(limit=1000)
2. Use metadata filtering to reduce scope
3. Run on machine with more RAM
```

### Charts Not Showing
```
Solution:
1. Verify plotly is installed: pip install plotly
2. Clear Streamlit cache: streamlit cache clear
3. Refresh browser (Ctrl+R)
4. Run again: streamlit run db_visualizer_advanced.py
```

---

## 📚 Understanding the Architecture

```
Your Application Stack:
┌─────────────────────────────────────┐
│         Streamlit App (UI)          │
│  ├─ app.py (Main Chatbot)           │
│  ├─ db_visualizer.py (Basic Viz)    │
│  └─ db_visualizer_advanced.py (Adv) │
├─────────────────────────────────────┤
│      LangChain (Orchestration)      │
│  ├─ Document Processing             │
│  ├─ Embedding Pipeline              │
│  └─ RAG Chain Setup                 │
├─────────────────────────────────────┤
│    ChromaDB (Vector Database)       │
│  ├─ Document Collections            │
│  ├─ Vector Embeddings               │
│  └─ Similarity Index                │
├─────────────────────────────────────┤
│   Document Storage: data_adenomyosis/│
│  ├─ PDF Files (20+ research papers) │
│  ├─ Text Extraction & Chunking      │
│  └─ Metadata Enrichment             │
└─────────────────────────────────────┘
```

---

## 🎓 DataModel Explanation for Adenomyosis App

### Document Ingestion Process
```
1. PDF Load
   Input: PDF file (e.g., "jurnal_adenomyosis_2023.pdf")
   Tool: PyMuPDF
   Output: Raw text extraction

2. Text Chunking
   Input: Raw text
   Tool: RecursiveCharacterTextSplitter
   Output: Chunks (overlap=200 chars)
   Metadata: page number, source file

3. Embedding Generation
   Input: Text chunks
   Model: sentence-transformers/paraphrase-multilingual-MiniLM
   Output: Vector (384 dimensions)
   Language: Indonesian/English support

4. ChromaDB Storage
   Input: Vectors + Metadata
   Storage: SQLite + Vector Index
   Output: Searchable collection

5. Query Processing
   Input: User question
   Step 1: Embed user query
   Step 2: Search similar documents
   Step 3: Filter by metadata
   Step 4: Rank by relevance
   Output: Top-K documents

6. LLM Generation
   Input: Query + Context
   Model: Gemma 2 (via HuggingFace)
   Output: Generated answer + sources
```

### Metadata Scheme
```json
{
  "source": "jurnal_adenomyosis_2023.pdf",
  "page": 5,
  "category": "Research",
  "year": 2023,
  "chunk_id": "chunk_001",
  "chunk_index": 1,
  "validity": "high"
}
```

---

## 💡 Advanced Tips

### Tip 1: Query Custom Collections
```python
# In a custom script or Streamlit app
import chromadb

client = chromadb.PersistentClient(
    path="./chroma_db_adenomyosis"
)

collection = client.get_collection(name="default")

# Custom query
results = collection.query(
    query_texts=["adenomyosis treatment options"],
    where={"year": {"$gte": 2020}},
    n_results=5
)

print(results)
```

### Tip 2: Export & Backup
```bash
# Backup database
Rcopy chroma_db_adenomyosis chroma_db_backup

# In PowerShell:
Copy-Item -Recurse chroma_db_adenomyosis chroma_db_backup
```

### Tip 3: Monitor Performance
```python
import time

# In ChromaDB queries:
start = time.time()
results = collection.query(...)
duration = time.time() - start

print(f"Query time: {duration:.3f}s")
# Good: < 1 second
# Acceptable: 1-5 seconds
# Slow: > 5 seconds
```

---

## 📞 Support & Resources

- **ChromaDB Docs**: https://docs.trychroma.com/
- **Streamlit Docs**: https://docs.streamlit.io/
- **LangChain Docs**: https://python.langchain.com/
- **Plotly Docs**: https://plotly.com/python/

---

## ✨ Version History

**v1.0 - April 2026**
- Initial release
- Basic visualizer (db_visualizer.py)
- Advanced visualizer (db_visualizer_advanced.py)
- Documentation & guides

---

**Last Updated**: April 15, 2026  
**Status**: ✅ Production Ready

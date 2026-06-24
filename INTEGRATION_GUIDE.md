# 🔗 Integration Guide - Add Database Visualizer to Your App

Panduan lengkap untuk mengintegrasikan Database Visualizer ke aplikasi utama Anda.

---

## 📁 Option 1: Standalone (Simple - Recommended untuk awal)

### Setup
Struktur folder tetap seperti sekarang:
```
chatbot_endo_adenom/
├── app.py                         (main aplikasi)
├── db_visualizer.py               (NEW - basic viz)
├── db_visualizer_advanced.py      (NEW - advanced viz)
├── chatbot.py
├── visualizations.py
└── ...
```

### Cara Menjalankan
```bash
# Terminal 1: Main App
streamlit run app.py

# Terminal 2: Basic Visualizer  
streamlit run db_visualizer.py

# Terminal 3 (optional): Advanced Visualizer
streamlit run db_visualizer_advanced.py
```

**Keuntungan**:
- ✅ Simple dan cepat setup
- ✅ Tidak perlu modifikasi app.py existing
- ✅ Mudah debug apabila ada error
- ✅ Bisa run terpisah

**Kerugian**:
- ❌ Multiple browser tabs
- ❌ Tidak terintegrasi seamlessly

---

## 📁 Option 2: Multipage Streamlit App (Professional)

### Step 1: Create Pages Directory
```bash
# Pastikan app.py ada di root
cd chatbot_endo_adenom

# Create pages folder
mkdir pages
```

### Step 2: Create Page Files

#### Step 2a: Rename app.py → pages/01_💬_Chat.py
```bash
# Copy app.py ke pages folder dengan nama baru
copy app.py "pages/01_💬_Chat.py"

# Buat minimal struktur di app.py baru
# (atau biarkan kosong untuk redirect)
```

#### Step 2b: Create pages/02_📊_Visualisasi.py
```python
"""
Halaman visualisasi data adenomyosis.
Menampilkan grafik, statistik, dan diagram.
"""

import streamlit as st

st.set_page_config(
    page_title="Visualisasi Data",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Visualisasi Data Adenomyosis")
st.markdown("Grafik dan statistik tentang adenomyosis.")

st.divider()

# Import dari file utama
from visualizations import (
    show_adenomyosis_diagram,
    show_symptoms_chart,
    show_treatment_options,
    show_age_distribution,
    show_adenomyosis_vs_endometriosis
)

# Buat tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Anatomi", "Gejala", "Pengobatan", "Umur", "Perbandingan"
])

with tab1: show_adenomyosis_diagram()
with tab2: show_symptoms_chart()
with tab3: show_treatment_options()
with tab4: show_age_distribution()
with tab5: show_adenomyosis_vs_endometriosis()
```

#### Step 2c: Create pages/03_🗄️_Database_Viz.py
```python
"""
Database Visualizer - Integrated page.
Menampilkan visualisasi dan analisis database ChromaDB.
"""

import streamlit as st

st.set_page_config(
    page_title="Database Visualizer",
    page_icon="🗄️",
    layout="wide"
)

st.title("🗄️ Database Visualizer")
st.markdown("Visualisasi struktur dan analisis database ChromaDB.")

st.divider()

# Pilih visualizer
viz_type = st.radio(
    "Choose Visualizer:",
    ["Basic Analysis", "Advanced Analysis"],
    horizontal=True
)

if viz_type == "Basic Analysis":
    from db_visualizer import (
        show_database_overview,
        show_documents_table,
        show_metadata_distribution,
        show_erd_diagram,
        show_adenomyosis_data_model
    )
    
    tab1, tab2, tab3 = st.tabs([
        "Overview", "Explorer", "Schema"
    ])
    
    with tab1:
        show_database_overview()
    
    with tab2:
        st.subheader("📁 Collections & Documents")
        from db_visualizer import get_chroma_client
        client = get_chroma_client()
        
        if client:
            collections = client.list_collections()
            if collections:
                col_names = [c.name for c in collections]
                sel = st.selectbox("Select collection:", col_names)
                
                sub_tab1, sub_tab2 = st.tabs(["Documents", "Metadata"])
                with sub_tab1:
                    show_documents_table(sel)
                with sub_tab2:
                    show_metadata_distribution(sel)
    
    with tab3:
        show_erd_diagram()

else:  # Advanced Analysis
    from db_visualizer_advanced import main as advanced_main
    advanced_main()
```

#### Step 2d: Create pages/04_📚_Informasi.py
```python
"""
FAQ dan informasi umum tentang adenomyosis.
"""

import streamlit as st
from common_questions import COMMON_QUESTIONS

st.set_page_config(
    page_title="Informasi",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Informasi & FAQ")
st.markdown("""
**Frequently Asked Questions** tentang Adenomyosis, Endometriosis, 
dan pengobatan menggunakan teknologi modern dan herbal.
""")

st.divider()

for q_data in COMMON_QUESTIONS.values():
    with st.expander(q_data["question"]):
        st.markdown(q_data["answer"])
```

### Step 3: Update Root app.py
```python
"""
Main Streamlit app entry point.
Handles initialization dan common configurations.
"""

import os
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")

import streamlit as st
from dotenv import load_dotenv

# Konfigurasi halaman (WAJIB di line pertama)
st.set_page_config(
    page_title="Asisten Adenomyosis",
    page_icon="🩺",
    layout="wide"
)

load_dotenv()

# Info: Streamlit akan auto-load pages dari folder 'pages/'
# Tidak perlu render di sini jika sudah membuat pages/

st.title("🩺 Asisten Ahli: Adenomyosis & Endometriosis")
st.markdown("""
Selamat datang! Gunakan navigasi di sidebar untuk:
- 💬 **Chat**: Tanya jawab dengan AI asisten
- 📊 **Visualisasi**: Lihat data dan grafik penelitian  
- 🗄️ **Database**: Analisis struktur database
- 📚 **Informasi**: Pertanyaan umum (FAQ)

Aplikasi ini menggunakan **RAG (Retrieval-Augmented Generation)** 
dengan database vektor **ChromaDB** untuk memberikan jawaban berbasis bukti ilmiah.
""")

st.divider()

st.info("""
**ℹ️ Disclaimer:** Informasi ini dikurasi dari database riset terbatas. 
Selalu konsultasikan keputusan medis dengan dokter spesialis (Obgyn).
""")
```

### Final Folder Structure
```
chatbot_endo_adenom/
├── app.py                                    (root entry point)
├── pages/
│   ├── 01_💬_Chat.py                        (original app.py)
│   ├── 02_📊_Visualisasi.py                 (viz page - NEW)
│   ├── 03_🗄️_Database_Viz.py                (db viz - NEW)
│   └── 04_📚_Informasi.py                   (faq page - NEW)
├── db_visualizer.py                         (basic visualization)
├── db_visualizer_advanced.py                (advanced visualization)
├── chatbot.py                                
├── visualizations.py
├── common_questions.py
├── evaluation.py
├── utilities.py
├── config.toml
├── requirements.txt ✅ (updated)
├── DATABASE_VISUALIZER_README.md ✅ (NEW)
├── QUICKSTART_GUIDE.md ✅ (NEW)
├── FEATURES_SUMMARY.md ✅ (NEW)
├── assets/
└── data_adenomyosis/
```

### Step 4: Run Integrated App
```bash
# Run main app (will auto-load all pages)
streamlit run app.py
```

**Expected Sidebar Navigation**:
```
🩺 Asisten Adenomyosis

💬 Chat
📊 Visualisasi  
🗄️ Database Viz
📚 Informasi
```

---

## 🔧 Implementation Guide Step-by-Step

### For Option 1 (Standalone):
```bash
# 1. Verify requirements installed
pip install -r requirements.txt

# 2. Test main app
streamlit run app.py

# 3. Test basic visualizer (new terminal)
streamlit run db_visualizer.py

# 4. Test advanced visualizer (new terminal)  
streamlit run db_visualizer_advanced.py
```

### For Option 2 (Multipage):
```bash
# 1. Create pages directory
mkdir pages

# 2. Create all page files (use scripts above)

# 3. Update root app.py (keep minimal)

# 4. Run app
streamlit run app.py

# 5. Click sidebar navigation items
```

---

## 🎯 Recommended: Option 2 (Multipage)

**Why Option 2 is Better**:
- ✅ Professional appearance
- ✅ Single browser window
- ✅ Better URL routing
- ✅ Shared session state
- ✅ Easier navigation  
- ✅ Better performance

---

## ⚙️ Configuration Options

### Control Visualizer Features
In `db_visualizer.py` or page files:

```python
# Limit data load for performance
DOCUMENT_LIMIT = 1000  # Was 5000

# Select which tabs to show
SHOW_TABS = ["Overview", "Explorer", "Schema"]

# Color scheme
COLOR_SCHEME = "reds"  # or "blues", "viridis", etc.

# Chart height
CHART_HEIGHT = 500
```

### Control Advanced Features
In `db_visualizer_advanced.py`:

```python
# Limit network graph nodes
MAX_METADATA_FIELDS = 5  # Per collection

# Timeline precision
GROUP_BY = "year"  # or "month", "quarter"

# Export options
EXPORT_FORMAT = ["json", "csv", "txt"]
```

---

## 🚨 Common Integration Issues & Fixes

### Issue 1: "Pages folder is empty"
```
Fix:
- Make sure files have names starting with numbers (01_, 02_)
- Put emoji/name after number: 01_💬_Chat.py
- Restart Streamlit
```

### Issue 2: "ModuleNotFoundError: cannot import"
```
Fix:
- Ensure all files in same directory as app.py
- Or add to Python path:
  import sys
  sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
- Restart kernel
```

### Issue 3: "Session state not shared between pages"
```
Fix:
- Define in root app.py before pages load:
  if 'key' not in st.session_state:
      st.session_state.key = value
```

### Issue 4: "Database not found when running pages"
```
Fix:
- Run main chat page first (loads database)
- Then navigate to database visualizer
- Or ensure chroma_db_adenomyosis/ has valid data
```

---

## 📝 Checklist for Integration

### Before Integration
- [ ] Verify db_visualizer.py works standalone: `streamlit run db_visualizer.py`
- [ ] Verify requirements.txt has plotly, altair, pandas
- [ ] Backup original app.py
- [ ] Test ChromaDB connection (run main app first)

### During Integration (Option 2)
- [ ] Create pages/ folder
- [ ] Copy files to pages/ with correct naming (01_, 02_, etc.)
- [ ] Update root app.py to minimal version
- [ ] Test each page individually
- [ ] Verify imports work correctly

### After Integration
- [ ] Test sidebar navigation
- [ ] Verify all pages load correctly
- [ ] Check performance (should be same or better)
- [ ] Verify database visualizer loads data
- [ ] Delete old files if needed

---

## 📊 Performance Tuning

If app is slow after integration:

```python
# In pages/03_🗄️_Database_Viz.py

# Option 1: Reduce limit
@st.cache_resource
def get_chroma_client():
    ...
    all_docs = collection.get(limit=1000)  # Reduced from 5000

# Option 2: Add caching
@st.cache_data(ttl=300)  # Cache 5 minutes
def get_collection_data(collection_name):
    ...
    return data

# Option 3: Filter metadata
results = collection.query(
    ...,
    where={"year": {"$gte": 2022}}  # Only recent docs
)
```

---

## 🎓 Tips & Best Practices

### Tip 1: Organize imports
```python
# At top of each page
from pathlib import Path
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Then import
from db_visualizer import get_chroma_client
from chatbot import ChatBot
```

### Tip 2: Use st.set_page_config once per page
```python
# Top of each page file
st.set_page_config(
    page_title="Page Title",
    page_icon="🎯",
    layout="wide"
)

# Rest of code
```

### Tip 3: Share resources across pages
```python
# In root app.py before pages load
@st.cache_resource
def get_chatbot():
    return ChatBot()

# Access in pages
bot = get_chatbot()
```

---

## 🔄 Migration Path

If already decided to migrate from standalone:

```
Current (Standalone):
app.py → run separate visualizer

After Migration (Multipage):
app.py/
  └─ pages/
      ├─ 01_Chat.py (was app.py)
      ├─ 02_Viz.py (was visualizations.py)
      └─ 03_Database.py (was db_visualizer.py)

Run: streamlit run app.py
```

---

## 📞 Support

**Issue**: Pages don't appear  
**Solution**: Restart Streamlit, check file naming (01_, 02_, etc.)

**Issue**: Import errors  
**Solution**: Use absolute imports, add to sys.path, verify file exists

**Issue**: Database not loading  
**Solution**: Run chat page first, verify chroma_db_adenomyosis/ exists

**Issue**: Slow performance  
**Solution**: Reduce document limit, add filtering, cache data

---

## 📚 References

- Streamlit Multi-page Apps: https://docs.streamlit.io/library/get-started/multipage-apps
- Streamlit Best Practices: https://docs.streamlit.io/library/get-started/main-concepts

---

**Version**: 1.0  
**Status**: ✅ Ready  
**Last Updated**: April 15, 2026

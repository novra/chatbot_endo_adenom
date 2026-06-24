# 📊 Database Visualizer - Features Summary

## 🎯 Apa Itu Database Visualizer?

Database Visualizer adalah suite aplikasi untuk **visualisasi dan analisis** database ChromaDB yang menyimpan dokumen-dokumen adenomyosis dalam bentuk vector embeddings.

### Karakteristik Utama
- ✅ **Real-time Analysis**: Analisis langsung dari database yang aktif
- ✅ **Interactive Charts**: Grafik interaktif dengan hover details
- ✅ **Multiple Perspectives**: Lihat data dari berbagai sudut pandang
- ✅ **Network Visualization**: Diagram hubungan antar entities
- ✅ **Timeline Analysis**: Tracking dokumen berdasarkan tahun publikasi
- ✅ **Metadata Exploration**: Deep dive ke metadata fields
- ✅ **Export Capabilities**: Download data untuk analisis lebih lanjut

---

## 📦 Apa Disertakan?

### 1. **db_visualizer.py** (Basic)
File utama untuk visualisasi dasar
```
4 Halaman Utama:
├─ Database Overview    → Dashboard dengan metrics
├─ Collections & Docs   → Browse documents & metadata
├─ Schema & Model       → Entity relationship diagram
└─ Statistics           → Collection statistics & charts
```

**Ukuran Kode**: ~450 lines  
**Dependencies**: streamlit, pandas, altair, chromadb  
**Performance**: Fast (< 1 second load)

### 2. **db_visualizer_advanced.py** (Advanced)
File untuk visualisasi advanced dengan Plotly
```
5 Interactive Tabs:
├─ Network Graph        → Collection relationship network
├─ Timeline             → Documents over years
├─ Metadata Correlation → Co-occurrence heatmap
├─ Size Distribution    → Pie chart per collection
└─ Content Analysis     → Metadata word frequency
```

**Ukuran Kode**: ~500 lines  
**Dependencies**: streamlit, plotly, pandas, chromadb  
**Performance**: Detailed (2-5 seconds untuk data besar)

### 3. **Documentation Files**
```
├─ DATABASE_VISUALIZER_README.md  (Dokumentasi komprehensif)
├─ QUICKSTART_GUIDE.md            (Setup & troubleshooting)
├─ FEATURES_SUMMARY.md            (File ini)
└─ requirements.txt               (Dependencies updated)
```

---

## 🚀 Quick Start (30 Detik)

### Langkah 1: Install Packages
```bash
pip install -r requirements.txt
```

### Langkah 2: Run Visualizer
```bash
# Option A: Basic Visualizer
streamlit run db_visualizer.py

# Option B: Advanced Visualizer
streamlit run db_visualizer_advanced.py
```

### Langkah 3: Open Browser
Browser akan otomatis membuka di `http://localhost:8501`

---

## 📊 Fitur Detail

### Feature Matrix

| Feature | Basic | Advanced | Notes |
|---------|-------|----------|-------|
| **Collection Metrics** | ✅ | ✅ | Total docs, collections count |
| **Document Browsing** | ✅ | ✅ | View document previews |
| **Metadata Analysis** | ✅ | ✅ | ✅ Advanced has heatmap |
| **Visual Charts** | ✅ (Altair) | ✅ (Plotly) | Plotly more interactive |
| **Network Diagram** | ❌ | ✅ | Shows collection relationships |
| **Timeline Chart** | ❌ | ✅ | Documents by publication year |
| **Correlation Matrix** | ❌ | ✅ | Metadata field correlations |
| **Data Export** | ❌ | ✅ | JSON & CSV export |
| **Performance** | Fast | Thorough | Trade-off: speed vs detail |

### Detailed Feature Descriptions

#### Collection Overview
**What**: Dashboard dengan ringkasan database  
**Shows**:
- Total number of collections  
- Total documents across all collections
- Document count per collection
- Metadata keys per collection
- Sample documents preview

**Use Case**: Cepat check health database

---

#### Documents Explorer
**What**: Browse individual documents  
**Shows**:
- Document ID
- Document preview (first 100 chars)
- All metadata fields
- Sortable/searchable table

**Use Case**: Find specific documents, verify content

---

#### Metadata Analysis
**What**: Deep dive into metadata structure  
**Shows**:
- Unique values per field
- Value distribution chart
- Top-N values ranking
- Frequency statistics

**Use Case**: Understand data patterns, quality checks

---

#### Network Visualization
**What**: Interactive graph of data structure  
**Shows**:
- ChromaDB central node (red)
- Collections as connected nodes (teal)
- Metadata fields as leaf nodes (light teal)  
- Relationship edges with connection strength

**Use Case**: Understand overall database architecture

---

#### Timeline Analysis
**What**: Document distribution over time  
**Shows**:
- X-axis: Publication year
- Y-axis: Document count
- Bar chart with frequency
- Year statistics (min, max, total)

**Use Case**: Verify temporal coverage of documents

---

#### Metadata Correlation
**What**: Which metadata fields occur together  
**Shows**:
- Heatmap matrix of field co-occurrence
- Blue intensity = correlation strength
- Helps identify field relationships

**Use Case**: Understand metadata dependencies

---

#### Size Distribution
**What**: How documents distributed across collections  
**Shows**:
- Pie chart with percentages
- Document count per collection
- Total summary statistics

**Use Case**: Verify balanced data distribution

---

#### Content Analysis
**What**: Frequency analysis of metadata values  
**Shows**:
- Bar chart of metadata values
- Source type frequency
- Document type distribution

**Use Case**: Identify prevalent document types

---

## 🔌 Integration dengan Main App

### Option 1: Standalone (Fast)
Jalankan visualizer terpisah:
```bash
# Terminal 1: Main app
streamlit run app.py

# Terminal 2: Visualizer
streamlit run db_visualizer.py
```

### Option 2: Embedded (Recommended)
Buat Streamlit multipage app:

**Struktur Folder** (Recommended):
```
chatbot_endo_adenom/
├── app.py                        # Main app
├── db_visualizer.py              # Basic viz
├── db_visualizer_advanced.py     # Advanced viz
├── pages/
│   ├── 01_💬_Chat.py            # Chat page
│   ├── 02_📊_Visualisasi.py     # Viz page
│   └── 03_🗄️_Database_Viz.py    # DB Viz page
├── chatbot.py
├── visualizations.py
└── utils.py
```

**pages/03_🗄️_Database_Viz.py** (Wrapper):
```python
import streamlit as st
from db_visualizer_advanced import main as advanced_main

st.set_page_config(
    page_title="Database Visualizer",
    page_icon="🗄️",
    layout="wide"
)

# Sidebar untuk pilih visualizer
viz_type = st.sidebar.radio(
    "Choose Visualizer:",
    ["Basic", "Advanced"]
)

if viz_type == "Basic":
    from db_visualizer import show_database_overview
    show_database_overview()
else:
    advanced_main()
```

---

## 📈 Performance Expectations

| Operation | Time | Notes |
|-----------|------|-------|
| Load visualizer | 2-3 sec | First load includes ChromaDB init |
| Database overview | 1 sec | Quick aggregation |
| Browse documents (1000) | 2-3 sec | Table rendering |
| Network graph | 3-5 sec | Graph calculation |
| Timeline analysis | 2-3 sec | Date aggregation |
| Create heatmap | 3-5 sec | Matrix calculation |

**Optimization Tips**:
- Reduce `limit` parameter untuk faster load
- Use metadata filtering untuk smaller dataset scope
- Run on machine with 8GB+ RAM untuk large collections

---

## 🎨 Visualization Types

### Altair Charts (Basic)
- **Bar Charts**: Category frequency  
- **Scatter Plots**: 2D distribution
- **Tooltips**: Hover details
- Fast rendering
- Good for exploratory analysis

### Plotly Charts (Advanced)  
- **Interactive**: Zoom, pan, select
- **Heatmaps**: Correlation matrices
- **Network graphs**: Relationship visualization
- **Pie charts**: Proportion display
- **Bar/timeline**: Temporal analysis

---

## 🔍 Key Metrics to Monitor

### Health Indicators
```
✅ Good Database State:
- Total Collections: 1+  
- Total Documents: 100+
- Avg Docs/Collection: 50+
- Metadata Coverage: 80%+
- Year Range: 2019-2024+

⚠️ Warning Signs:
- Documents: 50-100
- Metadata Coverage: 50-80%
- All docs same year

❌ Problem State:
- No collections found
- Total docs < 50
- No metadata
- Database corruption errors
```

### Quality Checks
```
From Timeline:
- Should see documents from multiple years
- Should NOT see all docs from single year

From Metadata Correlation:  
- Should see "source" field in 99%+ docs
- Should see "page" field in 80%+ docs

From Size Distribution:
- Should see relatively balanced distribution
- No single collection should be 99%+ of docs
```

---

## 🛠️ Customization Options

### Add Custom Visualization
In `db_visualizer_advanced.py`:

```python
def visualize_custom_analysis():
    """Your custom visualization."""
    st.subheader("🎨 Custom Analysis")
    
    # Your implementation here
    pass

# Add to main() tabs:
tab_custom = st.tabs([..., "Custom"])
with tab_custom[-1]:
    visualize_custom_analysis()
```

### Change Color Schemes
```python
# Altair
color=alt.Color(
    "Persentase:Q", 
    scale=alt.Scale(scheme="viridis")  # Change scheme
)

# Plotly
fig = px.bar(
    ...,
    color_continuous_scale="Turbo"  # Change scale
)
```

### Modify Data Limits
```python
# Reduce memory usage for large collections
collection.get(limit=500)  # Was 5000

# Or use metadata filtering
collection.query(
    ...,
    where={"year": {"$gte": 2022}}  # Only recent
)
```

---

## 📞 Troubleshooting Quick Reference

| Error | Quick Fix |
|-------|-----------|
| "No collections" | Run main app first (app.py) |
| "Connection failed" | Check chroma_db_adenomyosis/ exists |
| "Slow loading" | Reduce `limit` parameter |
| "Charts not showing" | pip install plotly, refresh browser |
| "Memory error" | Reduce document limit, use filters |

---

## 🎓 Learning Path

### Beginner
1. Run db_visualizer.py
2. Check Database Overview
3. Browse some documents
4. Read DATABASE_VISUALIZER_README.md

### Intermediate  
1. Run db_visualizer_advanced.py
2. Explore Timeline & Distribution
3. Check Metadata Correlation
4. Understand data model

### Advanced
1. Customize visualizations
2. Add new chart types
3. Integrate with main app
4. Export & analyze data

---

## 📚 Related Documentation

- **Setup Guide**: QUICKSTART_GUIDE.md
- **Full Docs**: DATABASE_VISUALIZER_README.md
- **Architecture**: See "data delivery pipeline" in README
- **Troubleshooting**: QUICKSTART_GUIDE.md → Troubleshooting section

---

## ✅ Checklist: Ready to Use

- [ ] Python 3.8+ installed
- [ ] requirements.txt installed
- [ ] ChromaDB database initialized (run app.py first)
- [ ] PDF files in data_adenomyosis/ folder
- [ ] Ports 8501 available
- [ ] 100+ MB free disk space
- [ ] Read QUICKSTART_GUIDE.md

---

## 🎉 What's Next?

After setting up visualizers:

1. **Analyze Your Data**
   - Check temporal distribution
   - Verify metadata completeness
   - Ensure balanced loading

2. **Optimize Performance**
   - Monitor query times
   - Adjust chunk sizes if needed
   - Use metadata filtering

3. **Extend Functionality**
   - Add custom analysis scripts
   - Create specific dashboards
   - Build automated reports

4. **Share Insights**
   - Export data for reports
   - Create static visualizations
   - Document findings

---

**Version**: 1.0  
**Status**: ✅ Ready to Use  
**Last Updated**: April 15, 2026

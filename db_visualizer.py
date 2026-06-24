"""
Database Visualizer untuk ChromaDB dan struktur aplikasi Adenomyosis Chatbot
Menampilkan statistik database, collections, documents, dan metadata
"""

import os
import streamlit as st
import pandas as pd
import altair as alt
import chromadb
from chromadb.config import Settings
from pathlib import Path
from datetime import datetime
import json
from collections import Counter
import plotly.graph_objects as go
import plotly.express as px


@st.cache_resource
def get_chroma_client():
    """Inisialisasi ChromaDB client."""
    try:
        persist_dir = Path("./chroma_db_adenomyosis")
        persist_dir.mkdir(parents=True, exist_ok=True)
        
        client = chromadb.PersistentClient(
            path=str(persist_dir),
        )
        return client
    except Exception as e:
        st.error(f"Error menginisialisasi ChromaDB: {e}")
        return None


def get_database_overview(client):
    """Dapatkan overview database ChromaDB."""
    try:
        collections = client.list_collections()
        
        overview = {
            "total_collections": len(collections),
            "collections": []
        }
        
        for collection in collections:
            try:
                doc_count = collection.count()
                metadata = collection.metadata if hasattr(collection, 'metadata') else {}
                
                # Ambil sample data untuk analisis
                all_docs = collection.get(limit=1000)
                
                collection_info = {
                    "name": collection.name,
                    "document_count": doc_count,
                    "metadata": metadata,
                    "sample_size": len(all_docs.get('ids', [])) if all_docs else 0,
                }
                
                overview["collections"].append(collection_info)
            except Exception as e:
                st.warning(f"Error mengakses collection {collection.name}: {e}")
                
        return overview
    except Exception as e:
        st.error(f"Error mendapatkan overview database: {e}")
        return None


def analyze_collection_documents(client, collection_name):
    """Analisis documents dalam collection."""
    try:
        collection = client.get_collection(name=collection_name)
        
        # Ambil semua documents (dengan limit)
        all_docs = collection.get(limit=5000)
        
        analysis = {
            "total_documents": len(all_docs.get('ids', [])),
            "ids": all_docs.get('ids', []),
            "metadatas": all_docs.get('metadatas', []),
            "documents": all_docs.get('documents', []),
            "embeddings_count": len(all_docs.get('embeddings', [])) if all_docs.get('embeddings') else 0,
        }
        
        # Analisis metadata
        if analysis["metadatas"]:
            metadata_keys = set()
            for meta in analysis["metadatas"]:
                if isinstance(meta, dict):
                    metadata_keys.update(meta.keys())
            analysis["metadata_keys"] = list(metadata_keys)
            
            # Analisis nilai metadata
            metadata_analysis = {}
            for key in analysis["metadata_keys"]:
                values = [meta.get(key) for meta in analysis["metadatas"] if isinstance(meta, dict)]
                metadata_analysis[key] = {
                    "unique_values": len(set(str(v) for v in values if v)),
                    "sample_values": list(set(str(v) for v in values if v))[:5]
                }
            analysis["metadata_analysis"] = metadata_analysis
        
        return analysis
    except Exception as e:
        st.error(f"Error menganalisis collection: {e}")
        return None


def show_database_overview():
    """Tampilkan overview database."""
    st.header("📊 Database Visualization & Analytics")
    
    client = get_chroma_client()
    if not client:
        return
    
    overview = get_database_overview(client)
    if not overview:
        return
    
    # Metrics row
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Collections", overview["total_collections"])
    with col2:
        total_docs = sum(c.get("document_count", 0) for c in overview["collections"])
        st.metric("Total Documents", total_docs)
    
    st.divider()
    
    # Collection details
    if overview["collections"]:
        st.subheader("📁 Database Collections")
        
        for idx, collection in enumerate(overview["collections"]):
            with st.expander(f"Collection: {collection['name']} ({collection['document_count']} documents)", expanded=idx==0):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Documents", collection["document_count"])
                with col2:
                    st.metric("Metadata Keys", len(collection.get("metadata", {})))
                with col3:
                    st.metric("Sample Size", collection["sample_size"])
                
                # Analisis detil
                analysis = analyze_collection_documents(client, collection['name'])
                
                if analysis:
                    st.write("**Metadata Keys:**", ", ".join(analysis.get("metadata_keys", [])))
                    
                    # Tampilkan metadata analysis
                    if analysis.get("metadata_analysis"):
                        st.write("**Metadata Analysis:**")
                        for key, stats in analysis["metadata_analysis"].items():
                            st.write(f"- **{key}**: {stats['unique_values']} unique values")
                            if stats['sample_values']:
                                st.caption(f"  Samples: {', '.join(stats['sample_values'][:3])}")
                    
                    # Tampilkan sample documents
                    st.write("**Sample Documents:**")
                    sample_df = pd.DataFrame({
                        "ID": analysis['ids'][:5],
                        "Document": [doc[:100] + "..." if len(doc) > 100 else doc 
                                   for doc in analysis['documents'][:5]]
                    })
                    st.dataframe(sample_df, use_container_width=True, hide_index=True)


def show_documents_table(collection_name):
    """Tampilkan tabel documents dari collection."""
    client = get_chroma_client()
    if not client:
        return
    
    try:
        collection = client.get_collection(name=collection_name)
        all_docs = collection.get(limit=1000)
        
        # Buat dataframe
        data = {
            "ID": all_docs.get('ids', [])[:50],
            "Document Preview": [doc[:80] + "..." if len(doc) > 80 else doc 
                               for doc in all_docs.get('documents', [])[:50]],
        }
        
        # Tambahkan metadata columns
        if all_docs.get('metadatas'):
            metadatas = all_docs.get('metadatas')[:50]
            if metadatas and isinstance(metadatas[0], dict):
                for key in metadatas[0].keys():
                    data[key] = [m.get(key, "-") if isinstance(m, dict) else "-" for m in metadatas]
        
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error menampilkan documents: {e}")


def show_metadata_distribution(collection_name):
    """Tampilkan distribusi metadata dalam chart."""
    client = get_chroma_client()
    if not client:
        return
    
    try:
        collection = client.get_collection(name=collection_name)
        all_docs = collection.get(limit=5000)
        metadatas = all_docs.get('metadatas', [])
        
        if not metadatas:
            st.info("Tidak ada metadata dalam collection ini.")
            return
        
        # Filter metadata yang berisi values
        valid_metadatas = [m for m in metadatas if isinstance(m, dict)]
        
        if not valid_metadatas:
            st.warning("Metadata tidak dalam format yang dapat dianalisis.")
            return
        
        # Dapatkan semua keys
        all_keys = set()
        for m in valid_metadatas:
            all_keys.update(m.keys())
        
        if not all_keys:
            st.info("Tidak ada metadata fields untuk divisualisasi.")
            return
        
        # Pilih metadata field untuk divisualisasi
        selected_field = st.selectbox("Pilih Metadata Field untuk Divisualisasi:", 
                                     list(all_keys), key=f"metadata_{collection_name}")
        
        if selected_field:
            # Kumpulkan values
            values = [str(m.get(selected_field, "unknown")) for m in valid_metadatas]
            value_counts = Counter(values)
            
            # Buat chart
            chart_data = pd.DataFrame({
                "Value": list(value_counts.keys()),
                "Count": list(value_counts.values())
            }).sort_values("Count", ascending=False).head(15)
            
            if len(chart_data) > 0:
                chart = alt.Chart(chart_data).mark_bar().encode(
                    x=alt.X("Count:Q", title="Jumlah"),
                    y=alt.Y("Value:N", title=selected_field, sort="-x")
                ).properties(
                    title=f"Distribusi: {selected_field}",
                    height=400
                )
                st.altair_chart(chart, use_container_width=True)
                
                # Tampilkan statistik
                st.write(f"**Total unique values:** {len(value_counts)}")
                st.write(f"**Top 5 values:**")
                for val, count in sorted(value_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                    st.write(f"- {val}: {count}")
            
    except Exception as e:
        st.error(f"Error menampilkan distribusi metadata: {e}")


def show_erd_diagram():
    """Tampilkan Entity-Relationship Diagram."""
    st.subheader("🗂️ Database Schema Diagram")
    
    erd_text = """
    ```
    CHROMADB SYSTEM
    ├── Collections
    │   ├── metadata
    │   └── embeddings (vector space)
    │
    ├── Documents
    │   ├── id (unique identifier)
    │   ├── content (text)
    │   ├── embedding (vector)
    │   ├── metadata (key-value pairs)
    │   └── timestamp
    │
    └── Query Operations
        ├── Similarity Search
        ├── Metadata Filtering
        └── Hybrid Search
    ```
    """
    
    st.markdown(erd_text)
    
    # Diagram visual
    st.write("**Komponen Utama:**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        **Collections**
        - Namespaces untuk documents
        - Menyimpan vector embeddings
        - Support metadata filtering
        """)
    
    with col2:
        st.warning("""
        **Documents**
        - Teks original
        - Numerical embeddings
        - Metadata attributes
        - Timestamps
        """)
    
    with col3:
        st.success("""
        **Query Operations**
        - Vector similarity
        - Metadata filters
        - Hybrid searches
        - Top-K retrieval
        """)


def show_adenomyosis_data_model():
    """Tampilkan data model untuk aplikasi Adenomyosis."""
    st.subheader("🩺 Adenomyosis Chatbot Data Model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Document Types:**")
        doc_types = {
            "Jurnal Penelitian": "Research papers about adenomyosis management",
            "Guidelines Klinis": "Clinical practice guidelines",
            "Review Herbal": "Herbal treatment reviews",
            "Panduan Penanganan": "Treatment guidelines"
        }
        
        for doc_type, desc in doc_types.items():
            st.markdown(f"- **{doc_type}**: {desc}")
    
    with col2:
        st.write("**Metadata Fields:**")
        metadata_fields = {
            "source": "PDF filename",
            "page": "Page number in document",
            "category": "Document category",
            "year": "Publication year",
            "chunk_id": "Text chunk identifier",
            "relevance": "Relevance score"
        }
        
        for field, desc in metadata_fields.items():
            st.markdown(f"- **{field}**: {desc}")
    
    # Data flow diagram
    st.write("**Data Flow:**")
    st.markdown("""
    ```
    PDF Documents (data_adenomyosis/)
         ↓
    Text Extraction (PyMuPDF)
         ↓
    Text Splitting (Recursive Chunks)
         ↓
    Embedding Generation (HuggingFace)
         ↓
    ChromaDB Storage (Vectors + Metadata)
         ↓
    RAG Query & Retrieval
         ↓
    LLM Generate Response
         ↓
    Display Answer
    ```
    """)


def main():
    """Main function untuk database visualizer."""
    st.set_page_config(
        page_title="Database Visualizer",
        page_icon="📊",
        layout="wide"
    )
    
    # Sidebar navigation
    with st.sidebar:
        st.title("📊 Database Visualizer")
        page = st.radio(
            "Pilih View:",
            ["Database Overview", "Collections & Documents", "Schema & Model", "Statistics"]
        )
    
    # Main content
    if page == "Database Overview":
        show_database_overview()
    
    elif page == "Collections & Documents":
        st.header("📁 Collections & Documents Explorer")
        
        client = get_chroma_client()
        if client:
            collections = client.list_collections()
            if collections:
                collection_names = [c.name for c in collections]
                selected_collection = st.selectbox("Pilih Collection:", collection_names)
                
                if selected_collection:
                    st.subheader(f"Collection: {selected_collection}")
                    
                    tab1, tab2 = st.tabs(["Documents Table", "Metadata Distribution"])
                    
                    with tab1:
                        show_documents_table(selected_collection)
                    
                    with tab2:
                        show_metadata_distribution(selected_collection)
            else:
                st.info("Tidak ada collections dalam database. Database mungkin kosong.")
    
    elif page == "Schema & Model":
        st.header("🗂️ Database Schema & Data Model")
        
        tab1, tab2 = st.tabs(["Schema Diagram", "Data Model"])
        
        with tab1:
            show_erd_diagram()
        
        with tab2:
            show_adenomyosis_data_model()
    
    elif page == "Statistics":
        st.header("📈 Database Statistics")
        
        client = get_chroma_client()
        if client:
            overview = get_database_overview(client)
            
            if overview and overview["collections"]:
                # Buat statistics
                stats_data = {
                    "Collection": [],
                    "Documents": [],
                    "Sample Size": []
                }
                
                for collection in overview["collections"]:
                    stats_data["Collection"].append(collection["name"])
                    stats_data["Documents"].append(collection["document_count"])
                    stats_data["Sample Size"].append(collection["sample_size"])
                
                df_stats = pd.DataFrame(stats_data)
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Collections", len(overview["collections"]))
                with col2:
                    st.metric("Total Documents", df_stats["Documents"].sum())
                with col3:
                    st.metric("Average Docs/Collection", 
                             int(df_stats["Documents"].sum() / len(overview["collections"])))
                
                st.divider()
                
                # Chart
                st.subheader("Documents per Collection")
                chart = alt.Chart(df_stats).mark_bar().encode(
                    x=alt.X("Collection:N", title="Collection Name"),
                    y=alt.Y("Documents:Q", title="Number of Documents")
                ).properties(height=400)
                st.altair_chart(chart, use_container_width=True)
                
                # Table
                st.subheader("Collection Statistics Table")
                st.dataframe(df_stats, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()

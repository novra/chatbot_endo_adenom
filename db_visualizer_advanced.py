"""
Advanced Database Visualization untuk ChromaDB
Menampilkan network diagrams, timeline analysis, dan interactive visualizations
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from collections import Counter
import chromadb
from pathlib import Path
import json


@st.cache_resource
def get_chroma_client():
    """Initialize ChromaDB client."""
    try:
        persist_dir = Path("./chroma_db_adenomyosis")
        persist_dir.mkdir(parents=True, exist_ok=True)
        
        client = chromadb.PersistentClient(
            path=str(persist_dir),
        )
        return client
    except Exception as e:
        st.error(f"Error initializing ChromaDB: {e}")
        return None


def visualize_collection_network():
    """
    Visualize collection structure sebagai network graph.
    Nodes: Collections, Metadata fields, Document count
    Edges: Relationships antara entities
    """
    st.subheader("🕸️ Collection Network Visualization")
    
    client = get_chroma_client()
    if not client:
        return
    
    try:
        collections = client.list_collections()
        
        if not collections:
            st.warning("No collections found")
            return
        
        # Build network data
        nodes = []
        edges = []
        
        # Central node
        nodes.append({
            'id': 'ChromaDB',
            'label': 'ChromaDB',
            'type': 'database',
            'size': 40
        })
        
        # Add collection nodes
        for collection in collections:
            doc_count = collection.count()
            nodes.append({
                'id': f'col_{collection.name}',
                'label': f'{collection.name}\n({doc_count} docs)',
                'type': 'collection',
                'size': 30,
                'value': doc_count
            })
            
            # Edge from ChromaDB to collection
            edges.append({
                'source': 'ChromaDB',
                'target': f'col_{collection.name}',
                'value': doc_count
            })
            
            # Get sample docs to analyze metadata
            try:
                sample = collection.get(limit=100)
                metadatas = sample.get('metadatas', [])
                
                # Collect unique metadata keys
                metadata_keys = set()
                for meta in metadatas:
                    if isinstance(meta, dict):
                        metadata_keys.update(meta.keys())
                
                # Add metadata field nodes
                for key in list(metadata_keys)[:5]:  # Limit to top 5
                    nodes.append({
                        'id': f'meta_{collection.name}_{key}',
                        'label': key,
                        'type': 'metadata',
                        'size': 20
                    })
                    
                    # Edge from collection to metadata field
                    edges.append({
                        'source': f'col_{collection.name}',
                        'target': f'meta_{collection.name}_{key}',
                        'value': 1
                    })
            except Exception as e:
                st.warning(f"Could not analyze metadata for {collection.name}: {e}")
        
        # Create Plotly figure
        fig = go.Figure()
        
        # Create node traces
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        node_color = []
        
        # Simple circular layout
        import math
        n_nodes = len(nodes)
        
        for i, node in enumerate(nodes):
            angle = 2 * math.pi * i / max(n_nodes, 1)
            x = 10 * math.cos(angle)
            y = 10 * math.sin(angle)
            
            if node['type'] == 'database':
                x, y = 0, 0  # Center
                color = '#FF6B6B'  # Red
            elif node['type'] == 'collection':
                color = '#4ECDC4'  # Teal
            else:  # metadata
                color = '#95E1D3'  # Light teal
            
            node_x.append(x)
            node_y.append(y)
            node_text.append(node['label'])
            node_size.append(node.get('size', 20))
            node_color.append(color)
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition="top center",
            hoverinfo='text',
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(width=2, color='white')
            ),
            textfont=dict(size=10)
        ))
        
        # Add edges (edges tidak divisualisasi keseluruhan untuk clarity)
        edge_x = []
        edge_y = []
        
        for edge in edges[:20]:  # Limit edges untuk clarity
            source_idx = next((i for i, n in enumerate(nodes) if n['id'] == edge['source']), None)
            target_idx = next((i for i, n in enumerate(nodes) if n['id'] == edge['target']), None)
            
            if source_idx is not None and target_idx is not None:
                edge_x.append([node_x[source_idx], node_x[target_idx], None])
                edge_y.append([node_y[source_idx], node_y[target_idx], None])
        
        if edge_x:
            for ex, ey in zip(edge_x, edge_y):
                fig.add_trace(go.Scatter(
                    x=ex, y=ey,
                    mode='lines',
                    line=dict(width=1, color='rgba(125,125,125,0.5)'),
                    hoverinfo='none',
                    showlegend=False
                ))
        
        fig.update_layout(
            title="ChromaDB Collection Network Structure",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='rgba(240,240,240,0.5)',
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Info box
        with st.expander("📋 Network Statistics"):
            st.write(f"**Total Nodes**: {len(nodes)}")
            st.write(f"**Total Collections**: {len([n for n in nodes if n['type'] == 'collection'])}")
            st.write(f"**Total Metadata Fields**: {len([n for n in nodes if n['type'] == 'metadata'])}")
    
    except Exception as e:
        st.error(f"Error creating network visualization: {e}")


def visualize_documents_timeline():
    """
    Timeline visualization of documents berdasarkan year metadata.
    """
    st.subheader("📅 Documents Timeline")
    
    client = get_chroma_client()
    if not client:
        return
    
    try:
        collections = client.list_collections()
        
        timeline_data = []
        
        for collection in collections:
            all_docs = collection.get(limit=5000)
            metadatas = all_docs.get('metadatas', [])
            
            # Extract years from metadata
            for meta in metadatas:
                if isinstance(meta, dict):
                    year = meta.get('year')
                    if year:
                        try:
                            year_int = int(year)
                            timeline_data.append({
                                'collection': collection.name,
                                'year': year_int,
                                'source': meta.get('source', 'Unknown')
                            })
                        except (ValueError, TypeError):
                            pass
        
        if timeline_data:
            df_timeline = pd.DataFrame(timeline_data)
            
            # Aggregate by year
            year_counts = df_timeline.groupby('year').size().reset_index(name='count')
            year_counts = year_counts.sort_values('year')
            
            # Create timeline chart
            fig = px.bar(
                year_counts,
                x='year',
                y='count',
                labels={'count': 'Number of Documents', 'year': 'Publication Year'},
                title='Document Distribution by Publication Year',
                color='count',
                color_continuous_scale='Viridis'
            )
            
            fig.update_xaxes(type='category')
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Earliest Year", int(year_counts['year'].min()))
            with col2:
                st.metric("Latest Year", int(year_counts['year'].max()))
            with col3:
                st.metric("Total Documents with Year", len(timeline_data))
        else:
            st.info("No year metadata found in documents")
            
    except Exception as e:
        st.error(f"Error creating timeline: {e}")


def visualize_metadata_correlation():
    """
    Heatmap visualization of metadata field correlations.
    """
    st.subheader("🔗 Metadata Field Correlation Matrix")
    
    client = get_chroma_client()
    if not client:
        return
    
    try:
        collection_names = st.selectbox(
            "Select collection for correlation analysis:",
            [c.name for c in client.list_collections()],
            key="corr_analysis"
        )
        
        if collection_names:
            collection = client.get_collection(name=collection_names)
            all_docs = collection.get(limit=2000)
            metadatas = all_docs.get('metadatas', [])
            
            # Build co-occurrence matrix
            valid_metadatas = [m for m in metadatas if isinstance(m, dict)]
            
            if valid_metadatas:
                # Get all unique keys
                all_keys = set()
                for meta in valid_metadatas:
                    all_keys.update(meta.keys())
                all_keys = sorted(list(all_keys))[:10]  # Limit to 10 keys
                
                # Build matrix
                matrix = pd.DataFrame(0, index=all_keys, columns=all_keys)
                
                for meta in valid_metadatas:
                    keys_present = [k for k in all_keys if k in meta and meta[k] is not None]
                    for key1 in keys_present:
                        for key2 in keys_present:
                            if key1 <= key2:  # Avoid duplicates
                                matrix.loc[key1, key2] += 1
                                if key1 != key2:
                                    matrix.loc[key2, key1] += 1
                
                # Create heatmap
                fig = px.imshow(
                    matrix,
                    labels=dict(color="Co-occurrence Count"),
                    title=f"Metadata Field Co-occurrence in {collection_names}",
                    color_continuous_scale="Blues"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No metadata found or invalid format")
                
    except Exception as e:
        st.error(f"Error creating correlation matrix: {e}")


def visualize_collection_size_distribution():
    """
    Pie chart menunjukkan distribusi document size antar collections.
    """
    st.subheader("📦 Collection Size Distribution")
    
    client = get_chroma_client()
    if not client:
        return
    
    try:
        collections = client.list_collections()
        
        collection_stats = []
        for collection in collections:
            doc_count = collection.count()
            collection_stats.append({
                'name': collection.name,
                'document_count': doc_count
            })
        
        df_stats = pd.DataFrame(collection_stats)
        
        if len(df_stats) > 1:
            fig = px.pie(
                df_stats,
                values='document_count',
                names='name',
                title='Document Distribution Across Collections',
                hole=0.3
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Only one or no collections found")
            
    except Exception as e:
        st.error(f"Error creating distribution chart: {e}")


def visualize_metadata_word_cloud():
    """
    Word frequency analysis dari metadata string fields.
    """
    st.subheader("☁️ Metadata Content Analysis")
    
    client = get_chroma_client()
    if not client:
        return
    
    try:
        collections = client.list_collections()
        if not collections:
            st.warning("No collections found")
            return
        
        collection = client.get_collection(name=collections[0].name)
        all_docs = collection.get(limit=1000)
        metadatas = all_docs.get('metadatas', [])
        
        # Extract source field for analysis
        sources = [m.get('source', '') for m in metadatas 
                  if isinstance(m, dict) and 'source' in m]
        
        if sources:
            # Count file types
            file_types = Counter([s.split('_')[-1] if '_' in s else s[:10] for s in sources])
            
            df_types = pd.DataFrame(
                list(file_types.items()),
                columns=['Type', 'Frequency']
            ).sort_values('Frequency', ascending=False)
            
            fig = px.bar(
                df_types,
                x='Type',
                y='Frequency',
                title='Document Source Type Frequency',
                color='Frequency',
                color_continuous_scale='Spectral'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No source metadata found")
            
    except Exception as e:
        st.error(f"Error creating word cloud analysis: {e}")


def main():
    """Main advanced visualizations page."""
    st.set_page_config(
        page_title="Advanced DB Visualization",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("🎨 Advanced Database Visualizations")
    st.markdown("""
    Comprehensive visual analysis tools untuk ChromaDB database structure dan content.
    """)
    
    # Tabs untuk berbagai visualisasi
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Network Graph",
        "Timeline",
        "Metadata Correlation",
        "Size Distribution",
        "Content Analysis"
    ])
    
    with tab1:
        visualize_collection_network()
    
    with tab2:
        visualize_documents_timeline()
    
    with tab3:
        visualize_metadata_correlation()
    
    with tab4:
        visualize_collection_size_distribution()
    
    with tab5:
        visualize_metadata_word_cloud()
    
    st.divider()
    
    # Export section
    st.subheader("💾 Data Export")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Export as JSON"):
            client = get_chroma_client()
            if client:
                export_data = {
                    "timestamp": datetime.now().isoformat(),
                    "collections": []
                }
                
                for collection in client.list_collections():
                    all_docs = collection.get(limit=1000)
                    export_data["collections"].append({
                        "name": collection.name,
                        "document_count": collection.count(),
                        "sample_documents": all_docs.get('ids', [])[:5]
                    })
                
                st.json(export_data)
    
    with col2:
        if st.button("Export Database Summary"):
            client = get_chroma_client()
            if client:
                summary = "DATABASE EXPORT SUMMARY\n"
                summary += "=" * 50 + "\n"
                summary += f"Exported: {datetime.now().isoformat()}\n\n"
                
                for collection in client.list_collections():
                    summary += f"Collection: {collection.name}\n"
                    summary += f"Documents: {collection.count()}\n"
                    summary += "-" * 30 + "\n"
                
                st.download_button(
                    label="Download Summary",
                    data=summary,
                    file_name=f"db_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )


if __name__ == "__main__":
    main()

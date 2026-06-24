#!/usr/bin/env python
"""
Debug script untuk mengatasi masalah ChromaDB schema incompatibility
Gunakan script ini jika ada error "no such column: collections.topic"
"""

import os
import sys
import shutil
import sqlite3
from pathlib import Path

def check_chromadb_version():
    """Check ChromaDB version yang terinstall."""
    try:
        import chromadb
        version = chromadb.__version__
        print(f"✅ ChromaDB version: {version}")
        return version
    except ImportError:
        print("❌ ChromaDB not installed")
        return None

def check_database_schema():
    """Periksa schema database ChromaDB."""
    db_path = Path("./chroma_db_adenomyosis/chroma.sqlite3")
    
    if not db_path.exists():
        print(f"⚠️ Database file not found: {db_path}")
        print("   Database belum dibuat atau path salah")
        return False
    
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Check if collections table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='collections';")
        if not cursor.fetchone():
            print("❌ 'collections' table not found in database")
            return False
        
        # Check columns in collections table
        cursor.execute("PRAGMA table_info(collections);")
        columns = cursor.fetchall()
        column_names = [col[1] for col in columns]
        
        print(f"✅ Collections table exists")
        print(f"   Columns: {', '.join(column_names)}")
        
        # Check if 'topic' column exists
        if 'topic' in column_names:
            print("✅ 'topic' column exists (ChromaDB 1.5.x schema)")
        else:
            print("⚠️ 'topic' column NOT found (likely older ChromaDB schema)")
        
        conn.close()
        return True
        
    except sqlite3.DatabaseError as e:
        print(f"❌ Database error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error checking schema: {e}")
        return False

def remove_old_database():
    """Hapus database lama yang tidak kompatibel."""
    db_path = Path("./chroma_db_adenomyosis")
    
    if not db_path.exists():
        print("✅ Database directory already removed or doesn't exist")
        return True
    
    try:
        print(f"🗑️ Removing database at {db_path}...")
        shutil.rmtree(str(db_path))
        print("✅ Database removed successfully")
        return True
    except Exception as e:
        print(f"❌ Error removing database: {e}")
        return False

def main():
    """Main debug function."""
    print("=" * 60)
    print("🔧 ChromaDB Schema Debug Tool")
    print("=" * 60)
    
    # Check ChromaDB version
    print("\n📦 Checking ChromaDB installation...")
    version = check_chromadb_version()
    
    # Check database schema
    print("\n📊 Checking database schema...")
    schema_ok = check_database_schema()
    
    # Recommendations
    print("\n" + "=" * 60)
    print("💡 RECOMMENDATIONS:")
    print("=" * 60)
    
    if not version:
        print("1. Install ChromaDB: pip install chromadb==0.4.22")
    else:
        major_version = int(version.split('.')[0])
        minor_version = int(version.split('.')[1])
        
        if major_version >= 1 or (major_version == 0 and minor_version >= 5):
            print(f"⚠️ ChromaDB {version} might have schema changes")
            print("   Recommended versions:")
            print("   - chromadb==0.4.22 (Stable, tested)")
            print("   - chromadb==0.4.24 (Also compatible)")
    
    if not schema_ok:
        print("\n2. Remove old database and rebuild:")
        print("   - Option A (Manual): Remove 'chroma_db_adenomyosis' folder")
        print("   - Option B (Script): Run this script with --cleanup flag")
        print("\n   Command: python debug_database.py --cleanup")
    else:
        print("\n✅ Database schema looks compatible")
    
    print("\n3. Verify requirements.txt has correct versions:")
    print("   chromadb==0.4.22")
    print("   langchain-chroma==0.1.4")
    
    print("\n4. Reinstall dependencies:")
    print("   pip install --force-reinstall -r requirements.txt")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--cleanup":
        print("🗑️ Starting database cleanup...")
        if remove_old_database():
            print("\n✅ Cleanup completed. Database will be rebuilt on next app run.")
        else:
            print("\n❌ Cleanup failed. Please manually remove 'chroma_db_adenomyosis' folder.")
    else:
        main()

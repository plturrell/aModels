#!/usr/bin/env python3
"""
Document ingestion utility for indexing corporate data into Elasticsearch.

Usage:
    python ingest.py --file documents.json --index corporate-kb
    python ingest.py --dir /path/to/docs --recursive --index corporate-kb
    python ingest.py --file policy.pdf --index corporate-kb
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from pypdf import PdfReader
from docx import Document

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DocumentIngester:
    def __init__(self, search_gateway_url: str = "http://localhost:8081"):
        self.search_gateway_url = search_gateway_url.rstrip("/")
        self.client = httpx.Client(timeout=30.0)

    def health_check(self) -> bool:
        """Check if search gateway is available"""
        try:
            response = self.client.get(f"{self.search_gateway_url}/health")
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    def parse_pdf(self, file_path: Path) -> str:
        """Extract text from PDF file"""
        try:
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error parsing PDF {file_path}: {e}")
            return ""

    def parse_docx(self, file_path: Path) -> str:
        """Extract text from DOCX file"""
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error parsing DOCX {file_path}: {e}")
            return ""

    def parse_txt(self, file_path: Path) -> str:
        """Extract text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            logger.error(f"Error parsing TXT {file_path}: {e}")
            return ""

    def parse_md(self, file_path: Path) -> str:
        """Extract text from Markdown file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            logger.error(f"Error parsing Markdown {file_path}: {e}")
            return ""

    def extract_text_from_file(self, file_path: Path) -> str:
        """Extract text from various file formats"""
        suffix = file_path.suffix.lower()
        
        if suffix == '.pdf':
            return self.parse_pdf(file_path)
        elif suffix == '.docx':
            return self.parse_docx(file_path)
        elif suffix == '.txt':
            return self.parse_txt(file_path)
        elif suffix in ['.md', '.markdown']:
            return self.parse_md(file_path)
        else:
            logger.warning(f"Unsupported file format: {suffix}")
            return ""

    def chunk_document(self, content: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split large documents into smaller chunks"""
        if len(content) <= chunk_size:
            return [content]
        
        chunks = []
        start = 0
        
        while start < len(content):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(content):
                # Look for sentence endings within the last 100 characters
                search_start = max(start + chunk_size - 100, start)
                for i in range(end - 1, search_start, -1):
                    if content[i] in '.!?':
                        end = i + 1
                        break
            
            chunk = content[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
        
        return chunks

    def create_document_metadata(self, file_path: Path, chunk_index: int = 0, total_chunks: int = 1) -> Dict[str, Any]:
        """Create metadata for document"""
        return {
            "filename": file_path.name,
            "filepath": str(file_path),
            "file_size": file_path.stat().st_size,
            "file_extension": file_path.suffix,
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "ingestion_timestamp": str(Path().cwd()),
        }

    def index_document_to_search_gateway(self, doc_id: str, content: str, metadata: Dict[str, Any]) -> bool:
        """Index document to search gateway"""
        try:
            document = {
                "id": doc_id,
                "content": content,
                "metadata": metadata
            }
            
            response = self.client.post(
                f"{self.search_gateway_url}/v1/documents",
                json=document
            )
            
            if response.status_code == 204:
                logger.info(f"Successfully indexed document: {doc_id}")
                return True
            else:
                logger.error(f"Failed to index document {doc_id}: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error indexing document {doc_id}: {e}")
            return False

    def ingest_file(self, file_path: Path, index_name: str = "corporate-kb") -> int:
        """Ingest a single file"""
        logger.info(f"Ingesting file: {file_path}")
        
        # Extract text content
        content = self.extract_text_from_file(file_path)
        if not content:
            logger.warning(f"No content extracted from {file_path}")
            return 0
        
        # Chunk large documents
        chunks = self.chunk_document(content)
        logger.info(f"Split document into {len(chunks)} chunks")
        
        indexed_count = 0
        for i, chunk in enumerate(chunks):
            doc_id = f"{file_path.stem}_{i}" if len(chunks) > 1 else file_path.stem
            metadata = self.create_document_metadata(file_path, i, len(chunks))
            metadata["index_name"] = index_name
            
            if self.index_document_to_search_gateway(doc_id, chunk, metadata):
                indexed_count += 1
        
        return indexed_count

    def ingest_directory(self, dir_path: Path, recursive: bool = False, index_name: str = "corporate-kb") -> int:
        """Ingest all documents in a directory"""
        logger.info(f"Ingesting directory: {dir_path}")
        
        if not dir_path.exists() or not dir_path.is_dir():
            logger.error(f"Directory does not exist: {dir_path}")
            return 0
        
        # Supported file extensions
        supported_extensions = {'.pdf', '.docx', '.txt', '.md', '.markdown'}
        
        # Find all files
        if recursive:
            files = [f for f in dir_path.rglob('*') if f.is_file() and f.suffix.lower() in supported_extensions]
        else:
            files = [f for f in dir_path.iterdir() if f.is_file() and f.suffix.lower() in supported_extensions]
        
        logger.info(f"Found {len(files)} files to ingest")
        
        total_indexed = 0
        for file_path in files:
            try:
                indexed_count = self.ingest_file(file_path, index_name)
                total_indexed += indexed_count
            except Exception as e:
                logger.error(f"Error ingesting {file_path}: {e}")
        
        return total_indexed

    def ingest_json_documents(self, json_file: Path, index_name: str = "corporate-kb") -> int:
        """Ingest documents from JSON file"""
        logger.info(f"Ingesting JSON documents: {json_file}")
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                documents = json.load(f)
        except Exception as e:
            logger.error(f"Error reading JSON file {json_file}: {e}")
            return 0
        
        if not isinstance(documents, list):
            logger.error("JSON file must contain a list of documents")
            return 0
        
        indexed_count = 0
        for i, doc in enumerate(documents):
            if not isinstance(doc, dict) or 'content' not in doc:
                logger.warning(f"Skipping invalid document at index {i}")
                continue
            
            doc_id = doc.get('id', f'doc_{i}')
            content = doc['content']
            metadata = doc.get('metadata', {})
            metadata['index_name'] = index_name
            metadata['source'] = 'json_import'
            
            if self.index_document_to_search_gateway(doc_id, content, metadata):
                indexed_count += 1
        
        return indexed_count


def main():
    parser = argparse.ArgumentParser(description='Ingest corporate documents into search index')
    parser.add_argument('--file', type=str, help='Single file to ingest')
    parser.add_argument('--dir', type=str, help='Directory to ingest')
    parser.add_argument('--recursive', action='store_true', help='Recursively search directories')
    parser.add_argument('--index', type=str, default='corporate-kb', help='Index name')
    parser.add_argument('--search-gateway-url', type=str, default='http://localhost:8081', 
                        help='Search gateway URL')
    
    args = parser.parse_args()
    
    if not args.file and not args.dir:
        parser.error("Must specify either --file or --dir")
    
    # Initialize ingester
    ingester = DocumentIngester(args.search_gateway_url)
    
    # Health check
    if not ingester.health_check():
        logger.error("Search gateway is not available. Please start the search gateway first.")
        sys.exit(1)
    
    total_indexed = 0
    
    try:
        if args.file:
            file_path = Path(args.file)
            if not file_path.exists():
                logger.error(f"File does not exist: {file_path}")
                sys.exit(1)
            
            if file_path.suffix.lower() == '.json':
                total_indexed = ingester.ingest_json_documents(file_path, args.index)
            else:
                total_indexed = ingester.ingest_file(file_path, args.index)
        
        elif args.dir:
            dir_path = Path(args.dir)
            total_indexed = ingester.ingest_directory(dir_path, args.recursive, args.index)
        
        logger.info(f"Successfully indexed {total_indexed} documents")
        
    except KeyboardInterrupt:
        logger.info("Ingestion interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

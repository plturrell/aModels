#!/usr/bin/env python3
"""
DMS to Gitea Migration Script

Migrates all documents from DMS PostgreSQL database to Extract service with Gitea storage.
Preserves metadata, catalog identifiers, and document versions.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import psycopg
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DMSMigrator:
    """Migrates documents from DMS to Extract service."""
    
    def __init__(
        self,
        dms_dsn: str,
        extract_url: str,
        gitea_url: Optional[str] = None,
        gitea_token: Optional[str] = None,
        project_id: str = "migrated",
        system_id: str = "dms-migration"
    ):
        self.dms_dsn = dms_dsn
        self.extract_url = extract_url.rstrip('/')
        self.gitea_url = gitea_url
        self.gitea_token = gitea_token
        self.project_id = project_id
        self.system_id = system_id
        self.mapping: Dict[str, Dict[str, Any]] = {}
        
        # Create HTTP session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
    
    def query_dms_documents(self) -> List[Dict[str, Any]]:
        """Query all documents from DMS PostgreSQL database."""
        logger.info("Connecting to DMS PostgreSQL database...")
        
        try:
            with psycopg.connect(self.dms_dsn) as conn:
                with conn.cursor() as cur:
                    # Query documents with their versions
                    cur.execute("""
                        SELECT 
                            d.id,
                            d.name,
                            d.description,
                            d.storage_path,
                            d.checksum,
                            d.catalog_identifier,
                            d.extraction_summary,
                            d.created_at,
                            d.updated_at,
                            COALESCE(
                                json_agg(
                                    json_build_object(
                                        'id', dv.id,
                                        'version_index', dv.version_index,
                                        'storage_path', dv.storage_path,
                                        'created_at', dv.created_at
                                    ) ORDER BY dv.version_index
                                ) FILTER (WHERE dv.id IS NOT NULL),
                                '[]'::json
                            ) as versions
                        FROM documents d
                        LEFT JOIN document_versions dv ON d.id = dv.document_id
                        GROUP BY d.id, d.name, d.description, d.storage_path, 
                                 d.checksum, d.catalog_identifier, d.extraction_summary,
                                 d.created_at, d.updated_at
                        ORDER BY d.created_at
                    """)
                    
                    documents = []
                    for row in cur.fetchall():
                        doc = {
                            'id': row[0],
                            'name': row[1],
                            'description': row[2],
                            'storage_path': row[3],
                            'checksum': row[4],
                            'catalog_identifier': row[5],
                            'extraction_summary': row[6],
                            'created_at': row[7].isoformat() if row[7] else None,
                            'updated_at': row[8].isoformat() if row[8] else None,
                            'versions': row[9] if isinstance(row[9], list) else json.loads(row[9]) if row[9] else []
                        }
                        documents.append(doc)
                    
                    logger.info(f"Found {len(documents)} documents in DMS database")
                    return documents
                    
        except Exception as e:
            logger.error(f"Failed to query DMS documents: {e}")
            raise
    
    def read_file(self, storage_path: str) -> Optional[bytes]:
        """Read file from DMS storage path."""
        try:
            path = Path(storage_path)
            if not path.exists():
                logger.warning(f"File not found: {storage_path}")
                return None
            
            with open(path, 'rb') as f:
                return f.read()
                
        except Exception as e:
            logger.error(f"Failed to read file {storage_path}: {e}")
            return None
    
    def upload_to_extract(
        self,
        file_data: bytes,
        filename: str,
        doc: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Upload document to Extract service."""
        try:
            # Prepare multipart form data
            files = {
                'file': (filename, file_data)
            }
            
            data = {
                'name': doc['name'],
                'description': doc.get('description', ''),
                'project_id': self.project_id,
                'system_id': self.system_id,
            }
            
            # Add Gitea storage config if provided
            if self.gitea_url and self.gitea_token:
                data['gitea_url'] = self.gitea_url
                data['gitea_token'] = self.gitea_token
                data['gitea_storage'] = json.dumps({
                    'enabled': True,
                    'owner': 'extract-service',
                    'repo_name': f'{self.project_id}-documents',
                    'base_path': 'documents/processed/',
                    'auto_create': True
                })
            
            # Upload to Extract service
            url = f"{self.extract_url}/documents/upload"
            logger.info(f"Uploading {filename} to Extract service...")
            
            response = self.session.post(
                url,
                files=files,
                data=data,
                timeout=300  # 5 minute timeout for large files
            )
            
            response.raise_for_status()
            result = response.json()
            
            logger.info(f"Successfully uploaded {filename}, document_id: {result.get('document_id')}")
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to upload {filename} to Extract service: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response: {e.response.text}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error uploading {filename}: {e}")
            return None
    
    def migrate_document(self, doc: Dict[str, Any]) -> bool:
        """Migrate a single document from DMS to Extract service."""
        doc_id = doc['id']
        logger.info(f"Migrating document {doc_id}: {doc['name']}")
        
        # Read file from storage path
        file_data = self.read_file(doc['storage_path'])
        if file_data is None:
            logger.warning(f"Skipping document {doc_id} - file not found")
            return False
        
        # Determine filename from storage path
        filename = Path(doc['storage_path']).name
        if not filename or filename == doc['storage_path']:
            # Fallback to document name
            filename = doc['name'] or f"document_{doc_id}"
        
        # Upload to Extract service
        result = self.upload_to_extract(file_data, filename, doc)
        if result is None:
            logger.error(f"Failed to migrate document {doc_id}")
            return False
        
        # Store mapping
        extract_doc_id = result.get('document_id')
        self.mapping[doc_id] = {
            'dms_id': doc_id,
            'extract_id': extract_doc_id,
            'dms_name': doc['name'],
            'dms_storage_path': doc['storage_path'],
            'gitea_url': result.get('gitea_url'),
            'catalog_identifier': doc.get('catalog_identifier'),
            'migrated_at': datetime.utcnow().isoformat(),
            'versions': doc.get('versions', [])
        }
        
        logger.info(f"Successfully migrated document {doc_id} -> {extract_doc_id}")
        return True
    
    def migrate_all(self, dry_run: bool = False) -> Dict[str, Any]:
        """Migrate all documents from DMS to Extract service."""
        logger.info("Starting DMS to Gitea migration...")
        logger.info(f"Extract service URL: {self.extract_url}")
        logger.info(f"Project ID: {self.project_id}")
        logger.info(f"System ID: {self.system_id}")
        
        if dry_run:
            logger.info("DRY RUN MODE - No documents will be migrated")
        
        # Query all documents
        documents = self.query_dms_documents()
        
        if not documents:
            logger.warning("No documents found in DMS database")
            return {
                'total': 0,
                'migrated': 0,
                'failed': 0,
                'mapping': {}
            }
        
        # Migrate each document
        migrated = 0
        failed = 0
        
        for i, doc in enumerate(documents, 1):
            logger.info(f"Processing document {i}/{len(documents)}: {doc['name']}")
            
            if dry_run:
                logger.info(f"DRY RUN: Would migrate {doc['id']}")
                migrated += 1
            else:
                if self.migrate_document(doc):
                    migrated += 1
                else:
                    failed += 1
        
        # Generate report
        report = {
            'total': len(documents),
            'migrated': migrated,
            'failed': failed,
            'mapping': self.mapping,
            'migration_date': datetime.utcnow().isoformat()
        }
        
        logger.info(f"Migration complete: {migrated} migrated, {failed} failed out of {len(documents)} total")
        
        return report
    
    def save_mapping(self, output_file: str):
        """Save migration mapping to JSON file."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.mapping, f, indent=2)
        
        logger.info(f"Migration mapping saved to {output_path}")


def main():
    """Main entry point for migration script."""
    parser = argparse.ArgumentParser(description='Migrate DMS documents to Extract service with Gitea storage')
    parser.add_argument(
        '--dms-dsn',
        default=os.getenv('DMS_POSTGRES_DSN'),
        help='DMS PostgreSQL connection string (default: DMS_POSTGRES_DSN env var)'
    )
    parser.add_argument(
        '--extract-url',
        default=os.getenv('EXTRACT_SERVICE_URL', 'http://localhost:8083'),
        help='Extract service URL (default: EXTRACT_SERVICE_URL env var or http://localhost:8083)'
    )
    parser.add_argument(
        '--gitea-url',
        default=os.getenv('GITEA_URL'),
        help='Gitea URL (default: GITEA_URL env var)'
    )
    parser.add_argument(
        '--gitea-token',
        default=os.getenv('GITEA_TOKEN'),
        help='Gitea token (default: GITEA_TOKEN env var)'
    )
    parser.add_argument(
        '--project-id',
        default=os.getenv('MIGRATION_PROJECT_ID', 'migrated'),
        help='Project ID for migrated documents (default: migrated)'
    )
    parser.add_argument(
        '--system-id',
        default=os.getenv('MIGRATION_SYSTEM_ID', 'dms-migration'),
        help='System ID for migrated documents (default: dms-migration)'
    )
    parser.add_argument(
        '--output',
        default='dms_migration_mapping.json',
        help='Output file for migration mapping (default: dms_migration_mapping.json)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Dry run mode - do not actually migrate documents'
    )
    
    args = parser.parse_args()
    
    # Validate required arguments
    if not args.dms_dsn:
        logger.error("DMS PostgreSQL DSN is required (--dms-dsn or DMS_POSTGRES_DSN env var)")
        sys.exit(1)
    
    # Create migrator
    migrator = DMSMigrator(
        dms_dsn=args.dms_dsn,
        extract_url=args.extract_url,
        gitea_url=args.gitea_url,
        gitea_token=args.gitea_token,
        project_id=args.project_id,
        system_id=args.system_id
    )
    
    # Run migration
    try:
        report = migrator.migrate_all(dry_run=args.dry_run)
        
        # Save mapping
        if not args.dry_run:
            migrator.save_mapping(args.output)
        
        # Print summary
        print("\n" + "="*60)
        print("MIGRATION SUMMARY")
        print("="*60)
        print(f"Total documents: {report['total']}")
        print(f"Migrated: {report['migrated']}")
        print(f"Failed: {report['failed']}")
        print(f"Migration date: {report['migration_date']}")
        print("="*60)
        
        if report['failed'] > 0:
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Migration failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()


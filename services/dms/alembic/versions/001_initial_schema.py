"""Initial schema

Revision ID: 001
Revises: 
Create Date: 2024-11-11 05:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create initial document management tables."""
    # Create documents table
    op.create_table(
        'documents',
        sa.Column('id', sa.String(36), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('storage_path', sa.String(512), nullable=False),
        sa.Column('checksum', sa.String(128), nullable=True),
        sa.Column('catalog_identifier', sa.String(255), nullable=True),
        sa.Column('extraction_summary', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('storage_path')
    )
    
    # Create index on created_at for efficient listing
    op.create_index('ix_documents_created_at', 'documents', ['created_at'], unique=False)
    
    # Create index on catalog_identifier for lookups
    op.create_index('ix_documents_catalog_identifier', 'documents', ['catalog_identifier'], unique=False)
    
    # Create document_versions table
    op.create_table(
        'document_versions',
        sa.Column('id', sa.String(36), nullable=False),
        sa.Column('document_id', sa.String(36), nullable=False),
        sa.Column('version_index', sa.Integer(), nullable=False),
        sa.Column('storage_path', sa.String(512), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create index on document_id for efficient version lookups
    op.create_index('ix_document_versions_document_id', 'document_versions', ['document_id'], unique=False)
    
    # Create composite index on document_id and version_index
    op.create_index('ix_document_versions_document_version', 'document_versions', ['document_id', 'version_index'], unique=True)


def downgrade() -> None:
    """Drop all tables."""
    op.drop_index('ix_document_versions_document_version', table_name='document_versions')
    op.drop_index('ix_document_versions_document_id', table_name='document_versions')
    op.drop_table('document_versions')
    
    op.drop_index('ix_documents_catalog_identifier', table_name='documents')
    op.drop_index('ix_documents_created_at', table_name='documents')
    op.drop_table('documents')

package piqa

import (
	"context"
	"database/sql"
	"encoding/binary"
	"fmt"
	"math"

	_ "github.com/SAP/go-hdb/driver"
)

// EmbeddingStorage provides efficient binary storage for embeddings using HANA
type EmbeddingStorage interface {
	SaveVectors(ctx context.Context, vectors map[string][]float32) error
	LoadVectors(ctx context.Context) (map[string][]float32, error)
	SaveContextEmbeddings(ctx context.Context, embeddings map[string]*ContextEmbeddings) error
	LoadContextEmbeddings(ctx context.Context) (map[string]*ContextEmbeddings, error)
	Close() error
}

// HANAEmbeddingStorage implements EmbeddingStorage using SAP HANA
type HANAEmbeddingStorage struct {
	db *sql.DB
}

// NewHANAEmbeddingStorage creates a new HANA-backed embedding storage
func NewHANAEmbeddingStorage(db *sql.DB) (*HANAEmbeddingStorage, error) {
	storage := &HANAEmbeddingStorage{db: db}
	if err := storage.initTables(); err != nil {
		return nil, fmt.Errorf("init tables: %w", err)
	}
	return storage, nil
}

// initTables creates necessary HANA tables
func (s *HANAEmbeddingStorage) initTables() error {
	queries := []string{
		`CREATE COLUMN TABLE IF NOT EXISTS PIQA_EMBEDDINGS (
			ID NVARCHAR(200) PRIMARY KEY,
			EMBEDDING_TYPE NVARCHAR(50) NOT NULL,
			VECTOR BLOB NOT NULL,
			DIMENSION INTEGER NOT NULL,
			CREATED_AT TIMESTAMP DEFAULT CURRENT_UTCTIMESTAMP,
			UPDATED_AT TIMESTAMP DEFAULT CURRENT_UTCTIMESTAMP
		)`,
		`CREATE COLUMN TABLE IF NOT EXISTS PIQA_CONTEXT_EMBEDDINGS (
			CONTEXT_ID NVARCHAR(200) PRIMARY KEY,
			PARAGRAPH_ID NVARCHAR(200) NOT NULL,
			START_POS INTEGER NOT NULL,
			END_POS INTEGER NOT NULL,
			TEXT CLOB,
			VECTOR BLOB NOT NULL,
			DIMENSION INTEGER NOT NULL,
			CREATED_AT TIMESTAMP DEFAULT CURRENT_UTCTIMESTAMP
		)`,
		`CREATE INDEX IF NOT EXISTS IDX_PIQA_EMB_TYPE ON PIQA_EMBEDDINGS(EMBEDDING_TYPE)`,
		`CREATE INDEX IF NOT EXISTS IDX_PIQA_CTX_PARA ON PIQA_CONTEXT_EMBEDDINGS(PARAGRAPH_ID)`,
	}

	for _, query := range queries {
		if _, err := s.db.Exec(query); err != nil {
			return fmt.Errorf("exec query: %w", err)
		}
	}

	return nil
}

// SaveVectors saves embeddings to HANA using efficient binary format
func (s *HANAEmbeddingStorage) SaveVectors(ctx context.Context, vectors map[string][]float32) error {
	stmt, err := s.db.PrepareContext(ctx,
		`UPSERT PIQA_EMBEDDINGS (ID, EMBEDDING_TYPE, VECTOR, DIMENSION) 
		 VALUES (?, 'question', ?, ?)`)
	if err != nil {
		return fmt.Errorf("prepare statement: %w", err)
	}
	defer stmt.Close()

	for id, vec := range vectors {
		vecBytes := serializeVector(vec)
		if _, err := stmt.ExecContext(ctx, id, vecBytes, len(vec)); err != nil {
			return fmt.Errorf("insert vector %s: %w", id, err)
		}
	}

	return nil
}

// LoadVectors loads embeddings from HANA
func (s *HANAEmbeddingStorage) LoadVectors(ctx context.Context) (map[string][]float32, error) {
	rows, err := s.db.QueryContext(ctx,
		"SELECT ID, VECTOR, DIMENSION FROM PIQA_EMBEDDINGS WHERE EMBEDDING_TYPE = 'question'")
	if err != nil {
		return nil, fmt.Errorf("query vectors: %w", err)
	}
	defer rows.Close()

	vectors := make(map[string][]float32)
	for rows.Next() {
		var id string
		var vecBytes []byte
		var dim int

		if err := rows.Scan(&id, &vecBytes, &dim); err != nil {
			return nil, fmt.Errorf("scan row: %w", err)
		}

		vectors[id] = deserializeVector(vecBytes, dim)
	}

	return vectors, nil
}

// SaveContextEmbeddings saves context embeddings to HANA
func (s *HANAEmbeddingStorage) SaveContextEmbeddings(ctx context.Context, embeddings map[string]*ContextEmbeddings) error {
	stmt, err := s.db.PrepareContext(ctx,
		`UPSERT PIQA_CONTEXT_EMBEDDINGS 
		 (CONTEXT_ID, PARAGRAPH_ID, START_POS, END_POS, TEXT, VECTOR, DIMENSION) 
		 VALUES (?, ?, ?, ?, ?, ?, ?)`)
	if err != nil {
		return fmt.Errorf("prepare statement: %w", err)
	}
	defer stmt.Close()

	for ctxID, emb := range embeddings {
		// Store each phrase embedding
		for i, phrase := range emb.Phrases {
			if i >= len(emb.Embeddings) {
				continue
			}
			vecBytes := serializeVector(emb.Embeddings[i])
			phraseID := fmt.Sprintf("%s_%d", ctxID, i)
			if _, err := stmt.ExecContext(ctx, phraseID, emb.ParagraphID, phrase.Start, phrase.End,
				phrase.Text, vecBytes, len(emb.Embeddings[i])); err != nil {
				return fmt.Errorf("insert context embedding %s: %w", phraseID, err)
			}
		}
	}

	return nil
}

// LoadContextEmbeddings loads context embeddings from HANA
func (s *HANAEmbeddingStorage) LoadContextEmbeddings(ctx context.Context) (map[string]*ContextEmbeddings, error) {
	rows, err := s.db.QueryContext(ctx,
		`SELECT CONTEXT_ID, PARAGRAPH_ID, START_POS, END_POS, TEXT, VECTOR, DIMENSION 
		 FROM PIQA_CONTEXT_EMBEDDINGS`)
	if err != nil {
		return nil, fmt.Errorf("query context embeddings: %w", err)
	}
	defer rows.Close()

	// Group by paragraph ID
	embeddingsByPara := make(map[string]*ContextEmbeddings)

	for rows.Next() {
		var ctxID, paraID, text string
		var start, end, dim int
		var vecBytes []byte

		if err := rows.Scan(&ctxID, &paraID, &start, &end, &text, &vecBytes, &dim); err != nil {
			return nil, fmt.Errorf("scan row: %w", err)
		}

		// Get or create ContextEmbeddings for this paragraph
		if _, exists := embeddingsByPara[paraID]; !exists {
			embeddingsByPara[paraID] = &ContextEmbeddings{
				ParagraphID: paraID,
				Phrases:     []Phrase{},
				Embeddings:  [][]float32{},
			}
		}

		// Add phrase and embedding
		embeddingsByPara[paraID].Phrases = append(embeddingsByPara[paraID].Phrases, Phrase{
			Text:  text,
			Start: start,
			End:   end,
		})
		embeddingsByPara[paraID].Embeddings = append(embeddingsByPara[paraID].Embeddings,
			deserializeVector(vecBytes, dim))
	}

	return embeddingsByPara, nil
}

// Close closes the database connection
func (s *HANAEmbeddingStorage) Close() error {
	return s.db.Close()
}

// serializeVector converts float32 slice to binary format
func serializeVector(vec []float32) []byte {
	buf := make([]byte, len(vec)*4)
	for i, v := range vec {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
	}
	return buf
}

// deserializeVector converts binary format to float32 slice
func deserializeVector(buf []byte, dim int) []float32 {
	vec := make([]float32, dim)
	for i := 0; i < dim && i*4 < len(buf); i++ {
		bits := binary.LittleEndian.Uint32(buf[i*4:])
		vec[i] = math.Float32frombits(bits)
	}
	return vec
}

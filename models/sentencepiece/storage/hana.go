package storage

import (
	"context"
	"database/sql"
	"fmt"

	_ "github.com/SAP/go-hdb/driver"
	"google.golang.org/protobuf/proto"

	pb "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Training/models/sentencepiece/internal/proto"
)

// HANAStorage provides model persistence using SAP HANA database.
type HANAStorage struct {
	db *sql.DB
}

// HANAConfig holds HANA connection configuration.
type HANAConfig struct {
	Host     string
	Port     int
	User     string
	Password string
	Database string
}

// NewHANAStorage creates a new HANA storage instance.
func NewHANAStorage(config *HANAConfig) (*HANAStorage, error) {
	// Build connection string
	connStr := fmt.Sprintf("hdb://%s:%s@%s:%d/%s",
		config.User,
		config.Password,
		config.Host,
		config.Port,
		config.Database,
	)

	// Open connection
	db, err := sql.Open("hdb", connStr)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to HANA: %w", err)
	}

	// Test connection
	if err := db.Ping(); err != nil {
		return nil, fmt.Errorf("failed to ping HANA: %w", err)
	}

	storage := &HANAStorage{db: db}

	// Initialize schema
	if err := storage.initSchema(); err != nil {
		return nil, fmt.Errorf("failed to initialize schema: %w", err)
	}

	return storage, nil
}

// initSchema creates the necessary tables if they don't exist.
func (h *HANAStorage) initSchema() error {
	schema := `
		CREATE TABLE IF NOT EXISTS sentencepiece_models (
			model_id NVARCHAR(255) PRIMARY KEY,
			model_name NVARCHAR(255) NOT NULL,
			model_type NVARCHAR(50) NOT NULL,
			vocab_size INTEGER NOT NULL,
			model_data BLOB NOT NULL,
			created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
			updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
			metadata NCLOB
		)
	`

	_, err := h.db.Exec(schema)
	return err
}

// SaveModel saves a model to HANA database.
func (h *HANAStorage) SaveModel(ctx context.Context, modelID string, modelName string, modelProto *pb.ModelProto) error {
	// Serialize model to protobuf bytes
	data, err := proto.Marshal(modelProto)
	if err != nil {
		return fmt.Errorf("failed to marshal model: %w", err)
	}

	// Determine model type
	modelType := "UNKNOWN"
	if modelProto.TrainerSpec != nil {
		switch modelProto.TrainerSpec.GetModelType() {
		case pb.TrainerSpec_UNIGRAM:
			modelType = "UNIGRAM"
		case pb.TrainerSpec_BPE:
			modelType = "BPE"
		case pb.TrainerSpec_WORD:
			modelType = "WORD"
		case pb.TrainerSpec_CHAR:
			modelType = "CHAR"
		}
	}

	vocabSize := len(modelProto.Pieces)

	// Insert or update model
	query := `
		UPSERT sentencepiece_models (model_id, model_name, model_type, vocab_size, model_data, updated_at)
		VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
		WITH PRIMARY KEY
	`

	_, err = h.db.ExecContext(ctx, query, modelID, modelName, modelType, vocabSize, data)
	if err != nil {
		return fmt.Errorf("failed to save model: %w", err)
	}

	return nil
}

// LoadModel loads a model from HANA database.
func (h *HANAStorage) LoadModel(ctx context.Context, modelID string) (*pb.ModelProto, error) {
	query := `
		SELECT model_data FROM sentencepiece_models WHERE model_id = ?
	`

	var data []byte
	err := h.db.QueryRowContext(ctx, query, modelID).Scan(&data)
	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("model not found: %s", modelID)
	}
	if err != nil {
		return nil, fmt.Errorf("failed to load model: %w", err)
	}

	// Deserialize protobuf
	var modelProto pb.ModelProto
	if err := proto.Unmarshal(data, &modelProto); err != nil {
		return nil, fmt.Errorf("failed to unmarshal model: %w", err)
	}

	return &modelProto, nil
}

// ListModels returns a list of all models in the database.
func (h *HANAStorage) ListModels(ctx context.Context) ([]ModelInfo, error) {
	query := `
		SELECT model_id, model_name, model_type, vocab_size, created_at, updated_at
		FROM sentencepiece_models
		ORDER BY updated_at DESC
	`

	rows, err := h.db.QueryContext(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("failed to list models: %w", err)
	}
	defer rows.Close()

	var models []ModelInfo
	for rows.Next() {
		var info ModelInfo
		err := rows.Scan(
			&info.ModelID,
			&info.ModelName,
			&info.ModelType,
			&info.VocabSize,
			&info.CreatedAt,
			&info.UpdatedAt,
		)
		if err != nil {
			return nil, fmt.Errorf("failed to scan model info: %w", err)
		}
		models = append(models, info)
	}

	return models, nil
}

// DeleteModel removes a model from the database.
func (h *HANAStorage) DeleteModel(ctx context.Context, modelID string) error {
	query := `DELETE FROM sentencepiece_models WHERE model_id = ?`

	result, err := h.db.ExecContext(ctx, query, modelID)
	if err != nil {
		return fmt.Errorf("failed to delete model: %w", err)
	}

	rows, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("failed to get rows affected: %w", err)
	}

	if rows == 0 {
		return fmt.Errorf("model not found: %s", modelID)
	}

	return nil
}

// UpdateMetadata updates the metadata for a model.
func (h *HANAStorage) UpdateMetadata(ctx context.Context, modelID string, metadata string) error {
	query := `
		UPDATE sentencepiece_models 
		SET metadata = ?, updated_at = CURRENT_TIMESTAMP
		WHERE model_id = ?
	`

	result, err := h.db.ExecContext(ctx, query, metadata, modelID)
	if err != nil {
		return fmt.Errorf("failed to update metadata: %w", err)
	}

	rows, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("failed to get rows affected: %w", err)
	}

	if rows == 0 {
		return fmt.Errorf("model not found: %s", modelID)
	}

	return nil
}

// Close closes the database connection.
func (h *HANAStorage) Close() error {
	if h.db != nil {
		return h.db.Close()
	}
	return nil
}

// ModelInfo contains metadata about a stored model.
type ModelInfo struct {
	ModelID   string
	ModelName string
	ModelType string
	VocabSize int
	CreatedAt string
	UpdatedAt string
}

//go:build !hana

package storage

import (
	"context"
	"fmt"
	"time"
)

func hanaUnsupported(feature string) error {
	return fmt.Errorf("%s requires building with the hana build tag", feature)
}

// HANASearchIndex stub implementation when HANA support is disabled.
type HANASearchIndex struct{}

func NewHANASearchIndex(string, *PrivacyConfig) (*HANASearchIndex, error) {
	return nil, hanaUnsupported("search index")
}

func (h *HANASearchIndex) AddDocument(context.Context, string, []float64, string, map[string]interface{}) error {
	return hanaUnsupported("AddDocument")
}

func (h *HANASearchIndex) SearchDocuments(context.Context, []float64, int) ([]SearchEmbedding, error) {
	return nil, hanaUnsupported("SearchDocuments")
}

func (h *HANASearchIndex) LogSearch(context.Context, string, []float64, int, string, string, string, int64) error {
	return hanaUnsupported("LogSearch")
}

func (h *HANASearchIndex) LogClick(context.Context, string, string, int, string) error {
	return hanaUnsupported("LogClick")
}

func (h *HANASearchIndex) Close() error {
	return nil
}

// HANASearchLogger stub implementation when HANA support is disabled.
type HANASearchLogger struct{}

func NewHANASearchLogger(string, *PrivacyConfig) (*HANASearchLogger, error) {
	return nil, hanaUnsupported("search logger")
}

func (h *HANASearchLogger) LogSearch(context.Context, *SearchLog) error {
	return hanaUnsupported("LogSearch")
}

func (h *HANASearchLogger) LogClick(context.Context, string, string, int, string) error {
	return hanaUnsupported("LogClick")
}

func (h *HANASearchLogger) GetSearchAnalytics(context.Context, time.Time, time.Time) (map[string]interface{}, error) {
	return nil, hanaUnsupported("GetSearchAnalytics")
}

func (h *HANASearchLogger) Close() error {
	return nil
}

// HANADocumentStore stub implementation when HANA support is disabled.
type HANADocumentStore struct{}

func NewHANADocumentStore(string, *PrivacyConfig) (*HANADocumentStore, error) {
	return nil, hanaUnsupported("document store")
}

func (h *HANADocumentStore) StoreDocument(context.Context, *Document) error {
	return hanaUnsupported("StoreDocument")
}

func (h *HANADocumentStore) GetDocument(context.Context, string, string, string) (*Document, error) {
	return nil, hanaUnsupported("GetDocument")
}

func (h *HANADocumentStore) UpdateDocument(context.Context, *Document, string, string) error {
	return hanaUnsupported("UpdateDocument")
}

func (h *HANADocumentStore) DeleteDocument(context.Context, string, string) error {
	return hanaUnsupported("DeleteDocument")
}

func (h *HANADocumentStore) ListDocuments(context.Context, string, int, int) ([]*Document, error) {
	return nil, hanaUnsupported("ListDocuments")
}

func (h *HANADocumentStore) SearchDocuments(context.Context, string, string, int) ([]*Document, error) {
	return nil, hanaUnsupported("SearchDocuments")
}

func (h *HANADocumentStore) GetDocumentStats(context.Context) (map[string]interface{}, error) {
	return nil, hanaUnsupported("GetDocumentStats")
}

func (h *HANADocumentStore) CleanupOldDocuments(context.Context) error {
	return hanaUnsupported("CleanupOldDocuments")
}

func (h *HANADocumentStore) Close() error {
	return nil
}

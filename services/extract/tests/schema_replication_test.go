package main

import (
	"context"
	"testing"

	sqlmock "github.com/DATA-DOG/go-sqlmock"
)

func TestReplicateSchemaToPostgres(t *testing.T) {
	db, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("sqlmock.New: %v", err)
	}
	defer db.Close()

	mock.ExpectExec("CREATE TABLE IF NOT EXISTS glean_nodes").WillReturnResult(sqlmock.NewResult(0, 0))
	mock.ExpectExec("CREATE TABLE IF NOT EXISTS glean_edges").WillReturnResult(sqlmock.NewResult(0, 0))
	mock.ExpectBegin()
	mock.ExpectPrepare("INSERT INTO glean_nodes").ExpectExec().WithArgs("node1", "table", "Table", sqlmock.AnyArg()).WillReturnResult(sqlmock.NewResult(1, 1))
	mock.ExpectPrepare("INSERT INTO glean_edges").ExpectExec().WithArgs("node1", "node2", "HAS_COLUMN", sqlmock.AnyArg()).WillReturnResult(sqlmock.NewResult(1, 1))
	mock.ExpectCommit()

	nodes := []Node{{ID: "node1", Type: "table", Label: "Table"}}
	edges := []Edge{{SourceID: "node1", TargetID: "node2", Label: "HAS_COLUMN"}}

	if err := replicateSchemaToPostgres(context.Background(), db, nodes, edges); err != nil {
		t.Fatalf("replicateSchemaToPostgres: %v", err)
	}

	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("postgres expectations: %v", err)
	}
}

func TestReplicateSchemaToHANA(t *testing.T) {
	db, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("sqlmock.New: %v", err)
	}
	defer db.Close()

	mock.ExpectExec("CREATE COLUMN TABLE IF NOT EXISTS GLEAN_NODES").WillReturnResult(sqlmock.NewResult(0, 0))
	mock.ExpectExec("CREATE COLUMN TABLE IF NOT EXISTS GLEAN_EDGES").WillReturnResult(sqlmock.NewResult(0, 0))
	mock.ExpectBegin()
	mock.ExpectPrepare("UPSERT GLEAN_NODES").ExpectExec().WithArgs("node1", "table", "Table", sqlmock.AnyArg()).WillReturnResult(sqlmock.NewResult(1, 1))
	mock.ExpectPrepare("UPSERT GLEAN_EDGES").ExpectExec().WithArgs("node1", "node2", "HAS_COLUMN", sqlmock.AnyArg()).WillReturnResult(sqlmock.NewResult(1, 1))
	mock.ExpectCommit()

	nodes := []Node{{ID: "node1", Type: "table", Label: "Table"}}
	edges := []Edge{{SourceID: "node1", TargetID: "node2", Label: "HAS_COLUMN"}}

	if err := replicateSchemaToHANA(context.Background(), db, nodes, edges); err != nil {
		t.Fatalf("replicateSchemaToHANA: %v", err)
	}

	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("hana expectations: %v", err)
	}
}

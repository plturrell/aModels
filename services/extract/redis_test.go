package main

import (
	"encoding/json"
	"testing"

	"github.com/alicebob/miniredis/v2"
	"github.com/redis/go-redis/v9"
)

func TestRedisPersistenceSaveSchema(t *testing.T) {
	mr, err := miniredis.Run()
	if err != nil {
		t.Fatalf("miniredis.Run: %v", err)
	}
	defer mr.Close()

	client := redis.NewClient(&redis.Options{Addr: mr.Addr()})
	rp := &RedisPersistence{client: client}

	nodes := []Node{{ID: "table1", Type: "table", Label: "Table"}}
	edges := []Edge{{SourceID: "table1", TargetID: "table1.col", Label: "HAS_COLUMN"}}

	if err := rp.SaveSchema(nodes, edges); err != nil {
		t.Fatalf("SaveSchema: %v", err)
	}

	nodeKey := "glean:nodes:table1"
	payload, err := mr.Get(nodeKey)
	if err != nil {
		t.Fatalf("expected node key %s: %v", nodeKey, err)
	}
	var stored map[string]any
	if err := json.Unmarshal([]byte(payload), &stored); err != nil {
		t.Fatalf("unmarshal payload: %v", err)
	}
	if stored["id"] != "table1" {
		t.Fatalf("unexpected node payload: %v", stored)
	}

	edgeKey := "glean:edges:table1:table1.col:HAS_COLUMN"
	if _, err := mr.Get(edgeKey); err != nil {
		t.Fatalf("expected edge key %s", edgeKey)
	}
}

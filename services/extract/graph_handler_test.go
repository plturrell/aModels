package main

import (
	"bytes"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

func TestHandleGraphProducesEnrichedMetadata(t *testing.T) {
	tmpDir := t.TempDir()

	jsonPath := filepath.Join(tmpDir, "orders.json")
	jsonPayload := `[
        {"order_id": 1, "amount": 13.37, "note": "urgent"},
        {"order_id": 2, "amount": 7.50},
        {"order_id": 3, "amount": null, "note": "gift"}
    ]`
	if err := os.WriteFile(jsonPath, []byte(jsonPayload), 0o600); err != nil {
		t.Fatalf("write json: %v", err)
	}

	controlMPath := filepath.Join(tmpDir, "jobs.xml")
	controlM := `<?xml version="1.0" encoding="UTF-8"?>
<JOBS>
  <JOB JOBNAME="JOB_ETL" COMMAND="run_etl.sh" RUN_AS="etl_user" APPLICATION="FIN" SUB_APPLICATION="AR" HOST="batch01" TASKTYPE="Command" CALENDAR_NAME="MONTH_END" TIMEFROM="0800" TIMETO="1700" CYCLIC="Y" INTERVAL="30" MAXRERUN="3" WHEN="AFTER" ODATE="20250131">
    <INCOND NAME="ETL.READY" ODATE="ODAT" SIGN="+" AND_OR="AND" />
    <OUTCOND NAME="ETL.DONE" ODATE="ODAT" SIGN="+" TYPE="P" />
  </JOB>
</JOBS>`
	if err := os.WriteFile(controlMPath, []byte(controlM), 0o600); err != nil {
		t.Fatalf("write control-m xml: %v", err)
	}

	server := &extractServer{
		logger:  log.New(io.Discard, "", 0),
		catalog: &Catalog{filePath: filepath.Join(tmpDir, "catalog.json")},
	}

	payload := graphRequest{
		JSONTables:    []string{jsonPath},
		ControlMFiles: []string{controlMPath},
		SqlQueries:    []string{"INSERT INTO public.tgt SELECT amount FROM public.src"},
	}

	body, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("marshal payload: %v", err)
	}

	req := httptest.NewRequest(http.MethodPost, "/graph", bytes.NewReader(body))
	rec := httptest.NewRecorder()

	server.handleGraph(rec, req)

	if rec.Result().StatusCode != http.StatusOK {
		t.Fatalf("expected status 200, got %d", rec.Result().StatusCode)
	}

	var resp struct {
		Nodes []Node `json:"nodes"`
		Edges []Edge `json:"edges"`
	}
	if err := json.NewDecoder(rec.Body).Decode(&resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}

	jobNode := findNode(resp.Nodes, "control-m:JOB_ETL")
	if jobNode == nil {
		t.Fatalf("expected control-m job node in graph")
	}
	schedule, ok := jobNode.Props["schedule"].(map[string]any)
	if !ok {
		t.Fatalf("expected schedule metadata on job node, got %#v", jobNode.Props)
	}
	if schedule["time_from"] != "08:00" || schedule["cyclic"] != true {
		t.Fatalf("unexpected schedule metadata: %#v", schedule)
	}
	orderDate, ok := schedule["order_date"].(map[string]any)
	if !ok || orderDate["type"] != "fixed" {
		t.Fatalf("expected decoded order_date metadata, got %#v", schedule["order_date"])
	}

	blockingEdge := findEdge(resp.Edges, "control-m-cond:ETL.READY", "control-m:JOB_ETL", "BLOCKS")
	if blockingEdge == nil || blockingEdge.Props == nil || blockingEdge.Props["odate"] == nil {
		t.Fatalf("expected BLOCKS edge with ODATE metadata, got %#v", blockingEdge)
	}

	columnNode := findNode(resp.Nodes, "orders.json.amount")
	if columnNode == nil {
		t.Fatalf("expected JSON column node")
	}
	if columnNode.Props["type"] != "number" {
		t.Fatalf("expected inferred numeric type, got %#v", columnNode.Props["type"])
	}
	if _, ok := columnNode.Props["presence_ratio"].(float64); !ok {
		t.Fatalf("expected presence ratio metadata, got %#v", columnNode.Props)
	}

	dataFlow := findEdge(resp.Edges, "public.src", "public.tgt", "DATA_FLOW")
	if dataFlow == nil {
		dataFlow = findEdge(resp.Edges, "public.src.amount", "public.tgt.amount", "DATA_FLOW")
	}
	if dataFlow == nil {
		for i := range resp.Edges {
			if resp.Edges[i].Label == "DATA_FLOW" {
				dataFlow = &resp.Edges[i]
				break
			}
		}
	}
	if dataFlow == nil {
		t.Fatalf("expected DATA_FLOW edge between source/target tables or columns")
	}
}

func findNode(nodes []Node, id string) *Node {
	for i := range nodes {
		if nodes[i].ID == id {
			return &nodes[i]
		}
	}
	return nil
}

func findEdge(edges []Edge, source, target, label string) *Edge {
	for i := range edges {
		edge := &edges[i]
		if edge.SourceID == source && edge.TargetID == target && edge.Label == label {
			return edge
		}
	}
	return nil
}

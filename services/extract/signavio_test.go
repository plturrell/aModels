package main

import (
	"os"
	"path/filepath"
	"testing"
)

const sampleBPMN = `<?xml version="1.0" encoding="UTF-8"?>
<definitions xmlns="http://www.omg.org/spec/BPMN/20100524/MODEL" xmlns:bpmndi="http://www.omg.org/spec/BPMN/20100524/DI">
  <process id="Process_Example" name="Example Process">
    <startEvent id="StartEvent_1" name="Start" />
    <task id="Task_1" name="First Task" />
    <task id="Task_2" name="Second Task" />
    <endEvent id="EndEvent_1" name="End" />
    <sequenceFlow id="Flow_1" sourceRef="StartEvent_1" targetRef="Task_1"/>
    <sequenceFlow id="Flow_2" sourceRef="Task_1" targetRef="Task_2"/>
    <sequenceFlow id="Flow_3" sourceRef="Task_2" targetRef="EndEvent_1"/>
  </process>
</definitions>`

func TestLoadSignavioArtifactsBPMN(t *testing.T) {
	dir := t.TempDir()
	file := filepath.Join(dir, "example.bpmn")
	if err := os.WriteFile(file, []byte(sampleBPMN), 0o644); err != nil {
		t.Fatalf("failed to write temp BPMN: %v", err)
	}

	nodes, edges, metadata := loadSignavioArtifacts([]string{file}, nil)

	if metadata.ProcessCount != 1 {
		t.Fatalf("expected 1 process, got %d", metadata.ProcessCount)
	}
	if len(nodes) == 0 || len(edges) == 0 {
		t.Fatalf("expected nodes and edges to be populated, got %d nodes / %d edges", len(nodes), len(edges))
	}
	if metadata.Processes[0].Name != "Example Process" {
		t.Fatalf("expected process name 'Example Process', got %q", metadata.Processes[0].Name)
	}
}

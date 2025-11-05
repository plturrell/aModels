package main

import "testing"

const sampleBPMN = `<?xml version="1.0" encoding="UTF-8"?>
<definitions xmlns="http://www.omg.org/spec/BPMN/20100524/MODEL">
  <process id="Process_1" name="Sample Onboarding">
    <laneSet>
      <lane id="Lane_1" name="Operations">
        <flowNodeRef>StartEvent_1</flowNodeRef>
        <flowNodeRef>Task_Approve</flowNodeRef>
        <flowNodeRef>EndEvent_1</flowNodeRef>
      </lane>
    </laneSet>
    <startEvent id="StartEvent_1" name="Start" />
    <task id="Task_Approve" name="Approve Request" />
    <endEvent id="EndEvent_1" name="Complete" />
    <sequenceFlow id="Flow_1" sourceRef="StartEvent_1" targetRef="Task_Approve" />
    <sequenceFlow id="Flow_2" sourceRef="Task_Approve" targetRef="EndEvent_1" />
  </process>
</definitions>`

func TestParseSignavioProcesses(t *testing.T) {
	models, err := parseSignavioProcesses([]byte(sampleBPMN))
	if err != nil {
		t.Fatalf("parseSignavioProcesses returned error: %v", err)
	}
	if len(models) != 1 {
		t.Fatalf("expected 1 model, got %d", len(models))
	}

	model := models[0]
	if model.ID != "Process_1" {
		t.Fatalf("unexpected process id %q", model.ID)
	}
	if model.Name != "Sample Onboarding" {
		t.Fatalf("unexpected process name %q", model.Name)
	}
	if len(model.Tasks) != 1 {
		t.Fatalf("expected 1 task, got %d", len(model.Tasks))
	}
	if model.Tasks[0].Name != "Approve Request" {
		t.Fatalf("unexpected task name %q", model.Tasks[0].Name)
	}
	if len(model.SequenceFlows) != 2 {
		t.Fatalf("expected 2 flows, got %d", len(model.SequenceFlows))
	}
}

func TestSignavioModelsToGraph(t *testing.T) {
	models, err := parseSignavioProcesses([]byte(sampleBPMN))
	if err != nil {
		t.Fatalf("parseSignavioProcesses returned error: %v", err)
	}

	nodes, edges := signavioModelsToGraph(models, "sample.bpmn")
	if len(nodes) == 0 {
		t.Fatalf("expected nodes to be produced")
	}

	var processFound bool
	var taskFound bool
	for _, node := range nodes {
		switch node.Type {
		case "signavio-process":
			processFound = true
		case "signavio-task":
			taskFound = true
		}
	}
	if !processFound {
		t.Fatalf("expected to find signavio-process node")
	}
	if !taskFound {
		t.Fatalf("expected to find signavio-task node")
	}

	var flowFound bool
	for _, edge := range edges {
		if edge.Label == "SIGNAVIO_FLOW" {
			flowFound = true
			break
		}
	}
	if !flowFound {
		t.Fatalf("expected to find SIGNAVIO_FLOW edge")
	}
}

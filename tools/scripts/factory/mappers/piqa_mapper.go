package mappers

import (
	"ai_benchmarks/scripts/factory"
	"fmt"
)

// PIQATask defines the structure for a PIQA physical reasoning task.
type PIQATask struct {
	Goal  string `json:"goal"`
	Sol1  string `json:"sol1"`
	Sol2  string `json:"sol2"`
	Label int    `json:"label"`
}

// PIQAMapper transforms data into physical commonsense reasoning tasks.
type PIQAMapper struct{}

func (m *PIQAMapper) Map(data factory.SourceData) ([]factory.BenchmarkTask, error) {
	row, ok := data.Content.(map[string]string)
	if !ok {
		return nil, fmt.Errorf("invalid data type for PIQAMapper")
	}

	goal := row["goal"]
	if goal == "" {
		goal = row["question"]
	}
	if goal == "" {
		return nil, fmt.Errorf("no goal field found")
	}

	sol1 := row["solution1"]
	sol2 := row["solution2"]

	if sol1 == "" || sol2 == "" {
		// Generate from single solution
		correctSol := row["solution"]
		if correctSol == "" {
			correctSol = row["answer"]
		}
		sol1 = correctSol
		sol2 = generateAlternativeSolution(correctSol)
	}

	label := 0
	if row["correct"] == "2" || row["label"] == "1" {
		label = 1
	}

	task := PIQATask{
		Goal:  goal,
		Sol1:  sol1,
		Sol2:  sol2,
		Label: label,
	}

	return []factory.BenchmarkTask{task}, nil
}

func generateAlternativeSolution(correct string) string {
	// Simple heuristic: create plausible but incorrect alternative
	return "Instead, " + correct
}

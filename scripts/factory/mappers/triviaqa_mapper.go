package mappers

import (
	"ai_benchmarks/scripts/factory"
	"fmt"
)

// TriviaQATask defines the structure for a TriviaQA task.
type TriviaQATask struct {
	Question string   `json:"question"`
	Answer   []string `json:"answer"`
	Evidence string   `json:"evidence,omitempty"`
}

// TriviaQAMapper transforms data into question-answering tasks.
type TriviaQAMapper struct{}

func (m *TriviaQAMapper) Map(data factory.SourceData) ([]factory.BenchmarkTask, error) {
	row, ok := data.Content.(map[string]string)
	if !ok {
		return nil, fmt.Errorf("invalid data type for TriviaQAMapper")
	}

	question := row["question"]
	if question == "" {
		question = row["query"]
	}
	if question == "" {
		return nil, fmt.Errorf("no question field found")
	}

	answer := row["answer"]
	if answer == "" {
		return nil, fmt.Errorf("no answer field found")
	}

	// TriviaQA supports multiple acceptable answers
	answers := []string{answer}
	if alt := row["answer_alias"]; alt != "" {
		answers = append(answers, alt)
	}

	evidence := row["evidence"]
	if evidence == "" {
		evidence = row["context"]
	}

	task := TriviaQATask{
		Question: question,
		Answer:   answers,
		Evidence: evidence,
	}

	return []factory.BenchmarkTask{task}, nil
}

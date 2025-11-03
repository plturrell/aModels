package mappers

import (
	"ai_benchmarks/scripts/factory"
	"fmt"
	"strings"
)

// BoolQTask defines the structure for a BoolQ question.
type BoolQTask struct {
	Question string `json:"question"`
	Answer   bool   `json:"answer"`
	Passage  string `json:"passage"`
}

// BoolQMapper transforms a data row into a True/False question.
type BoolQMapper struct{}

func (m *BoolQMapper) Map(data factory.SourceData) ([]factory.BenchmarkTask, error) {
	row, ok := data.Content.(map[string]string)
	if !ok {
		return nil, fmt.Errorf("invalid data type for BoolQMapper")
	}

	// Create a passage from the row data
	passage := fmt.Sprintf("In %s, the company %s had a revenue of $%s and a net income of $%s.",
		row["Year"], row["CompanyName"], row["Revenue"], row["NetIncome"])

	// Create a question based on the 'IsProfitable' column
	question := fmt.Sprintf("Based on the provided text, was %s profitable in the year %s?",
		row["CompanyName"], row["Year"])

	answer := strings.ToLower(row["IsProfitable"]) == "true"

	task := BoolQTask{
		Question: question,
		Answer:   answer,
		Passage:  passage,
	}

	return []factory.BenchmarkTask{task}, nil
}

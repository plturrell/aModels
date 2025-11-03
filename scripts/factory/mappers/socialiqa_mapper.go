package mappers

import (
	"ai_benchmarks/scripts/factory"
	"fmt"
)

// SocialIQATask defines the structure for a SocialIQA task.
type SocialIQATask struct {
	Context  string `json:"context"`
	Question string `json:"question"`
	AnswerA  string `json:"answerA"`
	AnswerB  string `json:"answerB"`
	AnswerC  string `json:"answerC"`
	Label    string `json:"label"`
}

// SocialIQAMapper transforms data into social reasoning tasks.
type SocialIQAMapper struct{}

func (m *SocialIQAMapper) Map(data factory.SourceData) ([]factory.BenchmarkTask, error) {
	row, ok := data.Content.(map[string]string)
	if !ok {
		return nil, fmt.Errorf("invalid data type for SocialIQAMapper")
	}

	context := row["context"]
	if context == "" {
		context = row["situation"]
	}

	question := row["question"]
	if question == "" {
		return nil, fmt.Errorf("no question field found")
	}

	answerA := row["answerA"]
	answerB := row["answerB"]
	answerC := row["answerC"]

	// If answers not provided, generate from single answer
	if answerA == "" {
		correctAnswer := row["answer"]
		answerA = correctAnswer
		answerB = "They would feel differently"
		answerC = "They would not care"
	}

	label := row["label"]
	if label == "" {
		label = "1" // Default to first answer
	}

	task := SocialIQATask{
		Context:  context,
		Question: question,
		AnswerA:  answerA,
		AnswerB:  answerB,
		AnswerC:  answerC,
		Label:    label,
	}

	return []factory.BenchmarkTask{task}, nil
}

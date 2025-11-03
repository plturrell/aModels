package arc

import (
	"fmt"
	"math"
)

// EvaluationMetrics tracks detailed performance metrics for ARC
type EvaluationMetrics struct {
	TotalTasks          int
	CorrectTasks        int
	PartialCredit       float64
	ByDifficulty        map[string]float64
	ByObjectComplexity  map[string]float64
	AverageAttempts     float64
	GeneralizationScore float64
}

// DifficultyLevel categorizes ARC tasks by complexity
type DifficultyLevel string

const (
	Easy     DifficultyLevel = "easy"
	Medium   DifficultyLevel = "medium"
	Hard     DifficultyLevel = "hard"
	VeryHard DifficultyLevel = "very_hard"
)

// TaskMetadata stores additional information about each task
type TaskMetadata struct {
	ID                  string
	Difficulty          DifficultyLevel
	ObjectCount         int
	GridSizeComplexity  float64
	RequiresAbstraction bool
	RequiresCounting    bool
	RequiresSymmetry    bool
}

// AnalyzeTaskDifficulty estimates the difficulty of an ARC task
func AnalyzeTaskDifficulty(task Task) TaskMetadata {
	metadata := TaskMetadata{
		GridSizeComplexity: calculateGridComplexity(task),
	}

	// Analyze training pairs
	if len(task.Train) > 0 {
		firstPair := task.Train[0]
		objects := DetectObjects(firstPair.In, 0)
		metadata.ObjectCount = len(objects)

		// Check for counting tasks
		metadata.RequiresCounting = isCountingTask(task.Train)

		// Check for symmetry requirements
		for _, obj := range objects {
			if obj.Symmetry != NoSymmetry {
				metadata.RequiresSymmetry = true
				break
			}
		}

		// Check if transformation is non-trivial
		metadata.RequiresAbstraction = !hasSimpleTransform(task.Train)
	}

	// Assign difficulty level
	metadata.Difficulty = assignDifficulty(metadata)

	return metadata
}

func calculateGridComplexity(task Task) float64 {
	complexity := 0.0
	count := 0

	for _, pair := range task.Train {
		h, w := len(pair.In), 0
		if h > 0 {
			w = len(pair.In[0])
		}
		complexity += float64(h * w)
		count++

		h, w = len(pair.Out), 0
		if h > 0 {
			w = len(pair.Out[0])
		}
		complexity += float64(h * w)
		count++
	}

	if count == 0 {
		return 0
	}
	return complexity / float64(count)
}

func hasSimpleTransform(pairs []Pair) bool {
	// Check if a simple color map works
	if _, ok := learnColorMap(pairs); ok {
		return true
	}

	// Check if a simple geometric transform works
	transforms := []func([][]int) [][]int{rot90, rot180, rot270, flipH, flipV}
	for _, tf := range transforms {
		works := true
		for _, p := range pairs {
			transformed := tf(p.In)
			if !equalGrid(transformed, p.Out) {
				works = false
				break
			}
		}
		if works {
			return true
		}
	}

	return false
}

func assignDifficulty(metadata TaskMetadata) DifficultyLevel {
	score := 0

	if metadata.ObjectCount > 5 {
		score += 2
	} else if metadata.ObjectCount > 2 {
		score += 1
	}

	if metadata.GridSizeComplexity > 400 {
		score += 2
	} else if metadata.GridSizeComplexity > 100 {
		score += 1
	}

	if metadata.RequiresAbstraction {
		score += 2
	}

	if metadata.RequiresCounting {
		score += 1
	}

	if metadata.RequiresSymmetry {
		score += 1
	}

	switch {
	case score >= 6:
		return VeryHard
	case score >= 4:
		return Hard
	case score >= 2:
		return Medium
	default:
		return Easy
	}
}

// EvaluateWithMetrics performs detailed evaluation with difficulty tracking
func EvaluateWithMetrics(tasks []Task, predictor func([][]int, []Pair) [][]int) EvaluationMetrics {
	metrics := EvaluationMetrics{
		ByDifficulty:       make(map[string]float64),
		ByObjectComplexity: make(map[string]float64),
	}

	difficultyCount := make(map[string]int)
	difficultyCorrect := make(map[string]int)

	for _, task := range tasks {
		metadata := AnalyzeTaskDifficulty(task)
		diffLevel := string(metadata.Difficulty)

		for _, testPair := range task.Test {
			metrics.TotalTasks++
			difficultyCount[diffLevel]++

			predicted := predictor(testPair.In, task.Train)
			if equalGrid(predicted, testPair.Out) {
				metrics.CorrectTasks++
				difficultyCorrect[diffLevel]++
			} else {
				// Partial credit based on similarity
				similarity := 1.0 - (GridDistance(predicted, testPair.Out) / float64(len(testPair.Out)*len(testPair.Out[0])))
				metrics.PartialCredit += similarity
			}
		}
	}

	// Calculate per-difficulty accuracy
	for diff, count := range difficultyCount {
		if count > 0 {
			metrics.ByDifficulty[diff] = float64(difficultyCorrect[diff]) / float64(count)
		}
	}

	// Calculate generalization score (weighted by difficulty)
	if metrics.TotalTasks > 0 {
		weights := map[string]float64{
			string(Easy):     1.0,
			string(Medium):   1.5,
			string(Hard):     2.0,
			string(VeryHard): 3.0,
		}

		weightedScore := 0.0
		totalWeight := 0.0

		for diff, acc := range metrics.ByDifficulty {
			weight := weights[diff]
			weightedScore += acc * weight
			totalWeight += weight
		}

		if totalWeight > 0 {
			metrics.GeneralizationScore = weightedScore / totalWeight
		}
	}

	return metrics
}

// PrintMetrics outputs detailed evaluation metrics
func PrintMetrics(metrics EvaluationMetrics) {
	fmt.Println("\n=== ARC Evaluation Metrics ===")
	fmt.Printf("Total Tasks: %d\n", metrics.TotalTasks)
	fmt.Printf("Correct: %d (%.2f%%)\n", metrics.CorrectTasks,
		100.0*float64(metrics.CorrectTasks)/float64(metrics.TotalTasks))
	fmt.Printf("Partial Credit Score: %.2f\n", metrics.PartialCredit)
	fmt.Printf("Generalization Score: %.2f\n", metrics.GeneralizationScore)

	fmt.Println("\nAccuracy by Difficulty:")
	for diff, acc := range metrics.ByDifficulty {
		fmt.Printf("  %s: %.2f%%\n", diff, 100.0*acc)
	}
}

// SplitDataset splits tasks into train/dev/test sets for proper evaluation
type DatasetSplit struct {
	Train []Task
	Dev   []Task
	Test  []Task
}

// CreateSplit divides tasks ensuring developer-aware generalization
func CreateSplit(tasks []Task, trainRatio, devRatio float64) DatasetSplit {
	n := len(tasks)
	nTrain := int(float64(n) * trainRatio)
	nDev := int(float64(n) * devRatio)

	// Sort by difficulty to ensure balanced splits
	taskMeta := make([]struct {
		task Task
		meta TaskMetadata
	}, n)

	for i, task := range tasks {
		taskMeta[i].task = task
		taskMeta[i].meta = AnalyzeTaskDifficulty(task)
	}

	// Stratified split by difficulty
	split := DatasetSplit{
		Train: make([]Task, 0, nTrain),
		Dev:   make([]Task, 0, nDev),
		Test:  make([]Task, 0, n-nTrain-nDev),
	}

	// Simple round-robin assignment by difficulty
	diffGroups := make(map[DifficultyLevel][]Task)
	for _, tm := range taskMeta {
		diffGroups[tm.meta.Difficulty] = append(diffGroups[tm.meta.Difficulty], tm.task)
	}

	for _, group := range diffGroups {
		nGroupTrain := int(math.Ceil(float64(len(group)) * trainRatio))
		nGroupDev := int(math.Ceil(float64(len(group)) * devRatio))

		split.Train = append(split.Train, group[:nGroupTrain]...)
		if nGroupTrain+nGroupDev <= len(group) {
			split.Dev = append(split.Dev, group[nGroupTrain:nGroupTrain+nGroupDev]...)
			split.Test = append(split.Test, group[nGroupTrain+nGroupDev:]...)
		} else {
			split.Dev = append(split.Dev, group[nGroupTrain:]...)
		}
	}

	return split
}

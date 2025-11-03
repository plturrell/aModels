package arc

import (
	"math/rand"
)

// ProgramSynthesis implements a more sophisticated search over the operation space
type ProgramSynthesis struct {
	MaxDepth        int
	MaxCandidates   int
	BackgroundColor int
	Rand            *rand.Rand
}

// Synthesize attempts to find a program that transforms inputs to outputs
func (ps *ProgramSynthesis) Synthesize(trainPairs []Pair) []Operation {
	if len(trainPairs) == 0 {
		return nil
	}

	// Build candidate operations based on analysis of training pairs
	candidates := ps.generateCandidateOperations(trainPairs)

	// Beam search over operation sequences
	bestProgram := []Operation{}
	bestScore := 0.0

	for depth := 1; depth <= ps.MaxDepth; depth++ {
		programs := ps.enumeratePrograms(candidates, depth)

		for _, program := range programs {
			score := ps.evaluateProgram(program, trainPairs)
			if score > bestScore {
				bestScore = score
				bestProgram = program
			}

			// Early exit if perfect match (within floating point precision)
			if score >= float64(len(trainPairs))-1e-6 {
				return program
			}
		}
	}

	return bestProgram
}

func (ps *ProgramSynthesis) generateCandidateOperations(trainPairs []Pair) []Operation {
	ops := []Operation{}

	// Always include basic geometric transforms
	ops = append(ops,
		&GeometricOp{Type: "rot90"},
		&GeometricOp{Type: "rot180"},
		&GeometricOp{Type: "rot270"},
		&GeometricOp{Type: "flipH"},
		&GeometricOp{Type: "flipV"},
	)

	// Analyze if object extraction is relevant
	hasObjects := false
	for _, p := range trainPairs {
		objects := DetectObjects(p.In, ps.BackgroundColor)
		if len(objects) > 1 {
			hasObjects = true
			break
		}
	}

	if hasObjects {
		ops = append(ops,
			&ExtractLargestObjectOp{BackgroundColor: ps.BackgroundColor},
			&AlignObjectsOp{Alignment: "top", BackgroundColor: ps.BackgroundColor},
			&AlignObjectsOp{Alignment: "left", BackgroundColor: ps.BackgroundColor},
		)
	}

	// Check if counting is relevant
	if isCountingTask(trainPairs) {
		for color := 0; color < 10; color++ {
			ops = append(ops, &CountColorOp{Color: color})
		}
	}

	// Check if pattern detection is relevant
	ops = append(ops,
		&DetectPatternOp{PatternType: "repeating"},
		&DetectPatternOp{PatternType: "symmetric"},
	)

	// Add goal inference
	ops = append(ops, &InferGoalOp{TrainPairs: trainPairs})

	return ops
}

func (ps *ProgramSynthesis) enumeratePrograms(ops []Operation, depth int) [][]Operation {
	if depth == 0 {
		return [][]Operation{{}}
	}

	if depth == 1 {
		programs := make([][]Operation, len(ops))
		for i, op := range ops {
			programs[i] = []Operation{op}
		}
		return programs
	}

	// Recursive enumeration with pruning
	subPrograms := ps.enumeratePrograms(ops, depth-1)
	programs := [][]Operation{}

	for _, subProg := range subPrograms {
		for _, op := range ops {
			newProg := make([]Operation, len(subProg)+1)
			copy(newProg, subProg)
			newProg[len(subProg)] = op
			programs = append(programs, newProg)

			// Limit candidates
			if len(programs) >= ps.MaxCandidates {
				return programs
			}
		}
	}

	return programs
}

func (ps *ProgramSynthesis) evaluateProgram(program []Operation, trainPairs []Pair) float64 {
	score := 0.0
	for _, pair := range trainPairs {
		predicted := pair.In
		for _, op := range program {
			predicted = op.Apply(predicted)
		}

		if equalGrid(predicted, pair.Out) {
			score += 1.0
		} else {
			// Partial credit for similarity
			score += 1.0 - (GridDistance(predicted, pair.Out) / float64(len(pair.Out)*len(pair.Out[0])))
		}
	}
	return score
}

// GeometricOp wraps the existing geometric transforms as Operations
type GeometricOp struct {
	Type string
}

func (op *GeometricOp) Apply(grid [][]int) [][]int {
	switch op.Type {
	case "rot90":
		return rot90(grid)
	case "rot180":
		return rot180(grid)
	case "rot270":
		return rot270(grid)
	case "flipH":
		return flipH(grid)
	case "flipV":
		return flipV(grid)
	}
	return grid
}

func (op *GeometricOp) Name() string {
	return op.Type
}

// ApplyProgram applies a synthesized program to a grid
func ApplyProgram(grid [][]int, program []Operation) [][]int {
	result := grid
	for _, op := range program {
		result = op.Apply(result)
	}
	return result
}

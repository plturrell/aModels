package socialiq

import (
	"fmt"
	"strings"
	"sync"
)

// ============================================================================
// Symbolic Reasoning Layer - Section 10 Implementation
// Implements neurosymbolic integration for explainable social reasoning
// ============================================================================

// LogicalPrimitive represents a basic logical concept
type LogicalPrimitive struct {
	Name      string
	Type      string // "predicate", "function", "constant"
	Arity     int
	Semantics string
	NeuralEmb []float32 // Neural embedding
}

// LogicalExpression represents a symbolic expression
type LogicalExpression struct {
	Operator string              // "AND", "OR", "NOT", "IMPLIES", "FORALL", "EXISTS"
	Args     []LogicalExpression // Recursive structure
	Atom     *LogicalAtom        // Leaf node
}

// LogicalAtom represents an atomic proposition
type LogicalAtom struct {
	Predicate string
	Terms     []string
	Negated   bool
}

// LogicalAxiom represents a logical rule or constraint
type LogicalAxiom struct {
	Name       string
	Premises   []LogicalExpression
	Conclusion LogicalExpression
	Confidence float64
}

// SymbolicMapper implements Section 10.1 - Neural-to-Symbolic Mapping
type SymbolicMapper struct {
	vocabulary map[string]LogicalPrimitive
	axioms     []LogicalAxiom
	mu         sync.RWMutex
}

// NewSymbolicMapper creates a new symbolic mapper
func NewSymbolicMapper() *SymbolicMapper {
	sm := &SymbolicMapper{
		vocabulary: make(map[string]LogicalPrimitive),
		axioms:     make([]LogicalAxiom, 0),
	}

	// Initialize with common social reasoning primitives
	sm.initializeSocialVocabulary()
	sm.initializeSocialAxioms()

	return sm
}

// initializeSocialVocabulary adds common social reasoning predicates
func (sm *SymbolicMapper) initializeSocialVocabulary() {
	primitives := []LogicalPrimitive{
		{Name: "Believes", Type: "predicate", Arity: 2, Semantics: "Agent X believes proposition P"},
		{Name: "Desires", Type: "predicate", Arity: 2, Semantics: "Agent X desires state S"},
		{Name: "Intends", Type: "predicate", Arity: 2, Semantics: "Agent X intends action A"},
		{Name: "Knows", Type: "predicate", Arity: 2, Semantics: "Agent X knows fact F"},
		{Name: "Feels", Type: "predicate", Arity: 2, Semantics: "Agent X feels emotion E"},
		{Name: "CausedBy", Type: "predicate", Arity: 2, Semantics: "Event E1 caused event E2"},
		{Name: "LeadsTo", Type: "predicate", Arity: 2, Semantics: "Action A leads to outcome O"},
		{Name: "SociallyAcceptable", Type: "predicate", Arity: 2, Semantics: "Action A is acceptable in context C"},
		{Name: "Empathizes", Type: "predicate", Arity: 2, Semantics: "Agent X empathizes with agent Y"},
		{Name: "Cooperates", Type: "predicate", Arity: 2, Semantics: "Agent X cooperates with agent Y"},
	}

	for _, prim := range primitives {
		sm.vocabulary[prim.Name] = prim
	}
}

// initializeSocialAxioms adds common social reasoning rules
func (sm *SymbolicMapper) initializeSocialAxioms() {
	// Axiom 1: Theory of Mind - If X believes Y desires Z, X may help Y achieve Z
	sm.axioms = append(sm.axioms, LogicalAxiom{
		Name: "TheoryOfMindHelping",
		Premises: []LogicalExpression{
			{Atom: &LogicalAtom{Predicate: "Believes", Terms: []string{"X", "Desires(Y, Z)"}}},
			{Atom: &LogicalAtom{Predicate: "Empathizes", Terms: []string{"X", "Y"}}},
		},
		Conclusion: LogicalExpression{
			Atom: &LogicalAtom{Predicate: "Intends", Terms: []string{"X", "Help(Y, Z)"}},
		},
		Confidence: 0.8,
	})

	// Axiom 2: Causal Reasoning - Actions lead to predictable outcomes
	sm.axioms = append(sm.axioms, LogicalAxiom{
		Name: "CausalAction",
		Premises: []LogicalExpression{
			{Atom: &LogicalAtom{Predicate: "Intends", Terms: []string{"X", "A"}}},
			{Atom: &LogicalAtom{Predicate: "LeadsTo", Terms: []string{"A", "O"}}},
		},
		Conclusion: LogicalExpression{
			Atom: &LogicalAtom{Predicate: "Believes", Terms: []string{"X", "WillHappen(O)"}},
		},
		Confidence: 0.9,
	})

	// Axiom 3: Social Norms - Socially acceptable actions are preferred
	sm.axioms = append(sm.axioms, LogicalAxiom{
		Name: "SocialNormPreference",
		Premises: []LogicalExpression{
			{Atom: &LogicalAtom{Predicate: "SociallyAcceptable", Terms: []string{"A", "C"}}},
			{Atom: &LogicalAtom{Predicate: "Desires", Terms: []string{"X", "O"}}},
			{Atom: &LogicalAtom{Predicate: "LeadsTo", Terms: []string{"A", "O"}}},
		},
		Conclusion: LogicalExpression{
			Atom: &LogicalAtom{Predicate: "Intends", Terms: []string{"X", "A"}},
		},
		Confidence: 0.85,
	})
}

// MapToSymbolic converts neural mental states to symbolic expressions
func (sm *SymbolicMapper) MapToSymbolic(state MentalState, agentID string) []LogicalExpression {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	expressions := make([]LogicalExpression, 0)

	// Map beliefs
	for belief, prob := range state.Beliefs {
		if prob > 0.5 { // Threshold for symbolic representation
			expressions = append(expressions, LogicalExpression{
				Atom: &LogicalAtom{
					Predicate: "Believes",
					Terms:     []string{agentID, belief},
					Negated:   false,
				},
			})
		}
	}

	// Map desires
	for desire, intensity := range state.Desires {
		if intensity > 0.5 {
			expressions = append(expressions, LogicalExpression{
				Atom: &LogicalAtom{
					Predicate: "Desires",
					Terms:     []string{agentID, desire},
					Negated:   false,
				},
			})
		}
	}

	// Map intentions
	for intention, commitment := range state.Intentions {
		if commitment > 0.5 {
			expressions = append(expressions, LogicalExpression{
				Atom: &LogicalAtom{
					Predicate: "Intends",
					Terms:     []string{agentID, intention},
					Negated:   false,
				},
			})
		}
	}

	// Map emotions
	for emotion, value := range state.Emotions {
		if value > 0.5 {
			expressions = append(expressions, LogicalExpression{
				Atom: &LogicalAtom{
					Predicate: "Feels",
					Terms:     []string{agentID, emotion},
					Negated:   false,
				},
			})
		}
	}

	return expressions
}

// SymbolicProver implements logical inference
type SymbolicProver struct {
	axioms        []LogicalAxiom
	knowledgeBase []LogicalExpression
	maxDepth      int
	mu            sync.RWMutex
}

// NewSymbolicProver creates a new symbolic prover
func NewSymbolicProver(axioms []LogicalAxiom) *SymbolicProver {
	return &SymbolicProver{
		axioms:        axioms,
		knowledgeBase: make([]LogicalExpression, 0),
		maxDepth:      5,
	}
}

// Prove attempts to prove a conclusion from premises
func (sp *SymbolicProver) Prove(premises []LogicalExpression, conclusion LogicalExpression) (bool, []LogicalAxiom, float64) {
	sp.mu.Lock()
	defer sp.mu.Unlock()

	// Add premises to knowledge base
	sp.knowledgeBase = append(sp.knowledgeBase, premises...)

	// Try to derive conclusion
	proof, confidence := sp.forwardChain(conclusion, 0)

	return len(proof) > 0, proof, confidence
}

// forwardChain performs forward chaining inference
func (sp *SymbolicProver) forwardChain(goal LogicalExpression, depth int) ([]LogicalAxiom, float64) {
	if depth > sp.maxDepth {
		return nil, 0.0
	}

	// Check if goal is already in knowledge base
	for _, expr := range sp.knowledgeBase {
		if sp.unify(expr, goal) {
			return []LogicalAxiom{}, 1.0
		}
	}

	// Try to apply axioms
	for _, axiom := range sp.axioms {
		if sp.unify(axiom.Conclusion, goal) {
			// Check if all premises are satisfied
			allSatisfied := true
			minConfidence := axiom.Confidence

			for _, premise := range axiom.Premises {
				subProof, conf := sp.forwardChain(premise, depth+1)
				if len(subProof) == 0 && conf == 0.0 {
					allSatisfied = false
					break
				}
				if conf < minConfidence {
					minConfidence = conf
				}
			}

			if allSatisfied {
				return []LogicalAxiom{axiom}, minConfidence
			}
		}
	}

	return nil, 0.0
}

// unify checks if two logical expressions can be unified
func (sp *SymbolicProver) unify(expr1, expr2 LogicalExpression) bool {
	// Simplified unification - just check atom predicates
	if expr1.Atom != nil && expr2.Atom != nil {
		return expr1.Atom.Predicate == expr2.Atom.Predicate
	}
	return false
}

// ============================================================================
// Hybrid Neurosymbolic Model - Section 10.2
// ============================================================================

// HybridNeuroSymbolicModel combines neural and symbolic reasoning
type HybridNeuroSymbolicModel struct {
	neuralEncoder  *NeuralEncoder
	symbolicMapper *SymbolicMapper
	symbolicProver *SymbolicProver
	fusionWeights  map[string]float64
	mu             sync.RWMutex
}

// NeuralEncoder encodes context into embeddings
type NeuralEncoder struct {
	embeddingDim int
}

// NewHybridNeuroSymbolicModel creates a new hybrid model
func NewHybridNeuroSymbolicModel() *HybridNeuroSymbolicModel {
	mapper := NewSymbolicMapper()

	return &HybridNeuroSymbolicModel{
		neuralEncoder:  &NeuralEncoder{embeddingDim: 768},
		symbolicMapper: mapper,
		symbolicProver: NewSymbolicProver(mapper.axioms),
		fusionWeights: map[string]float64{
			"neural":   0.6,
			"symbolic": 0.4,
		},
	}
}

// Predict combines neural and symbolic reasoning
func (hnsm *HybridNeuroSymbolicModel) Predict(input MultimodalInput) (PredictionResult, error) {
	hnsm.mu.Lock()
	defer hnsm.mu.Unlock()

	// Neural pathway
	neuralScores := hnsm.neuralPredict(input)

	// Symbolic pathway
	symbolicScores := hnsm.symbolicPredict(input)

	// Fusion
	finalScores := make([]float64, len(input.Answers))
	for i := range finalScores {
		finalScores[i] = hnsm.fusionWeights["neural"]*neuralScores[i] +
			hnsm.fusionWeights["symbolic"]*symbolicScores[i]
	}

	// Find best answer
	bestIdx := 0
	bestScore := finalScores[0]
	for i, score := range finalScores {
		if score > bestScore {
			bestScore = score
			bestIdx = i
		}
	}

	return PredictionResult{
		QuestionID:      input.VideoID + "_q",
		PredictedIndex:  bestIdx,
		PredictedAnswer: input.Answers[bestIdx],
		Confidence:      bestScore,
		Scores:          finalScores,
	}, nil
}

// neuralPredict performs neural prediction
func (hnsm *HybridNeuroSymbolicModel) neuralPredict(input MultimodalInput) []float64 {
	// Simplified neural prediction
	scores := make([]float64, len(input.Answers))

	// Simple keyword matching (in production, use actual neural model)
	for i, answer := range input.Answers {
		score := 0.5
		if strings.Contains(input.Transcript, answer) {
			score += 0.3
		}
		if strings.Contains(input.Question, answer) {
			score += 0.2
		}
		scores[i] = score
	}

	return scores
}

// symbolicPredict performs symbolic reasoning
func (hnsm *HybridNeuroSymbolicModel) symbolicPredict(input MultimodalInput) []float64 {
	scores := make([]float64, len(input.Answers))

	// Extract symbolic knowledge from context
	premises := hnsm.extractSymbolicPremises(input)

	// Score each answer based on logical entailment
	for i, answer := range input.Answers {
		conclusion := hnsm.answerToLogicalExpression(answer)

		// Try to prove the conclusion
		proved, proof, confidence := hnsm.symbolicProver.Prove(premises, conclusion)

		if proved {
			scores[i] = confidence
		} else {
			scores[i] = 0.3 // Base score for unprovable
		}

		// Bonus for using more axioms (deeper reasoning)
		scores[i] += float64(len(proof)) * 0.05
	}

	return scores
}

// extractSymbolicPremises extracts logical premises from input
func (hnsm *HybridNeuroSymbolicModel) extractSymbolicPremises(input MultimodalInput) []LogicalExpression {
	premises := make([]LogicalExpression, 0)

	// Simple extraction based on keywords
	transcript := strings.ToLower(input.Transcript)

	if strings.Contains(transcript, "happy") || strings.Contains(transcript, "excited") {
		premises = append(premises, LogicalExpression{
			Atom: &LogicalAtom{
				Predicate: "Feels",
				Terms:     []string{"speaker", "happiness"},
			},
		})
	}

	if strings.Contains(transcript, "want") || strings.Contains(transcript, "need") {
		premises = append(premises, LogicalExpression{
			Atom: &LogicalAtom{
				Predicate: "Desires",
				Terms:     []string{"speaker", "goal"},
			},
		})
	}

	if strings.Contains(transcript, "will") || strings.Contains(transcript, "going to") {
		premises = append(premises, LogicalExpression{
			Atom: &LogicalAtom{
				Predicate: "Intends",
				Terms:     []string{"speaker", "action"},
			},
		})
	}

	return premises
}

// answerToLogicalExpression converts an answer to a logical expression
func (hnsm *HybridNeuroSymbolicModel) answerToLogicalExpression(answer string) LogicalExpression {
	answer = strings.ToLower(answer)

	// Map common answer patterns to logical expressions
	if strings.Contains(answer, "happy") || strings.Contains(answer, "joy") {
		return LogicalExpression{
			Atom: &LogicalAtom{
				Predicate: "Feels",
				Terms:     []string{"speaker", "happiness"},
			},
		}
	}

	if strings.Contains(answer, "sad") || strings.Contains(answer, "upset") {
		return LogicalExpression{
			Atom: &LogicalAtom{
				Predicate: "Feels",
				Terms:     []string{"speaker", "sadness"},
			},
		}
	}

	// Default
	return LogicalExpression{
		Atom: &LogicalAtom{
			Predicate: "Unknown",
			Terms:     []string{answer},
		},
	}
}

// Train implements neurosymbolic training (Algorithm 7 from paper)
func (hnsm *HybridNeuroSymbolicModel) Train(dataset QADataset) error {
	// Simplified training - adjust fusion weights based on performance

	neuralCorrect := 0
	symbolicCorrect := 0
	total := 0

	for _, question := range dataset.Questions {
		input := MultimodalInput{
			VideoID:  question.VideoID,
			Question: question.Question,
			Answers:  question.Answers,
		}

		neuralScores := hnsm.neuralPredict(input)
		symbolicScores := hnsm.symbolicPredict(input)

		// Check which pathway got it right
		neuralPred := argmax(neuralScores)
		symbolicPred := argmax(symbolicScores)

		if neuralPred == question.CorrectIndex {
			neuralCorrect++
		}
		if symbolicPred == question.CorrectIndex {
			symbolicCorrect++
		}
		total++
	}

	// Adjust fusion weights based on accuracy
	if total > 0 {
		neuralAcc := float64(neuralCorrect) / float64(total)
		symbolicAcc := float64(symbolicCorrect) / float64(total)

		totalAcc := neuralAcc + symbolicAcc
		if totalAcc > 0 {
			hnsm.fusionWeights["neural"] = neuralAcc / totalAcc
			hnsm.fusionWeights["symbolic"] = symbolicAcc / totalAcc
		}
	}

	return nil
}

// Evaluate evaluates the hybrid model
func (hnsm *HybridNeuroSymbolicModel) Evaluate(dataset QADataset) (*EvaluationMetrics, error) {
	correct := 0
	total := len(dataset.Questions)

	for _, question := range dataset.Questions {
		input := MultimodalInput{
			VideoID:  question.VideoID,
			Question: question.Question,
			Answers:  question.Answers,
		}

		result, err := hnsm.Predict(input)
		if err != nil {
			continue
		}

		if result.PredictedIndex == question.CorrectIndex {
			correct++
		}
	}

	accuracy := float64(correct) / float64(total)

	return &EvaluationMetrics{
		Accuracy:           accuracy,
		TotalQuestions:     total,
		CorrectPredictions: correct,
		F1Score:            accuracy, // Simplified
		Precision:          accuracy,
		Recall:             accuracy,
	}, nil
}

// Helper function
func argmax(scores []float64) int {
	maxIdx := 0
	maxVal := scores[0]
	for i, val := range scores {
		if val > maxVal {
			maxVal = val
			maxIdx = i
		}
	}
	return maxIdx
}

// GenerateExplanation generates symbolic explanation for a prediction
func (hnsm *HybridNeuroSymbolicModel) GenerateExplanation(input MultimodalInput, prediction PredictionResult) string {
	premises := hnsm.extractSymbolicPremises(input)
	conclusion := hnsm.answerToLogicalExpression(input.Answers[prediction.PredictedIndex])

	proved, proof, confidence := hnsm.symbolicProver.Prove(premises, conclusion)

	explanation := fmt.Sprintf("Prediction: %s (Confidence: %.2f)\n\n",
		prediction.PredictedAnswer, prediction.Confidence)

	if proved {
		explanation += "Logical Proof:\n"
		for i, axiom := range proof {
			explanation += fmt.Sprintf("%d. %s (confidence: %.2f)\n",
				i+1, axiom.Name, axiom.Confidence)
		}
		explanation += fmt.Sprintf("\nOverall logical confidence: %.2f\n", confidence)
	} else {
		explanation += "No complete logical proof found. Relying on neural reasoning.\n"
	}

	explanation += fmt.Sprintf("\nFusion weights: Neural=%.2f, Symbolic=%.2f\n",
		hnsm.fusionWeights["neural"], hnsm.fusionWeights["symbolic"])

	return explanation
}

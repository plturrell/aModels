package socialiq

import (
	"context"
	"fmt"
	"strings"
	"sync"
)

// ============================================================================
// Multi-Agent Social Reasoning Framework
// Implements the Metacognitive Multi-Agent Architecture from Section 2.2.1
// ============================================================================

// TheoryOfMindAgent generates and maintains mental state distributions
type TheoryOfMindAgent struct {
	monoid          *MentalStateMonoid
	priorKnowledge  map[string]MentalState
	inferenceEngine InferenceEngine
	mu              sync.RWMutex
}

// NewTheoryOfMindAgent creates a new Theory of Mind agent
func NewTheoryOfMindAgent() *TheoryOfMindAgent {
	return &TheoryOfMindAgent{
		monoid:          NewMentalStateMonoid(),
		priorKnowledge:  make(map[string]MentalState),
		inferenceEngine: &BayesianInferenceEngine{},
	}
}

// InferMentalStates computes ψ(c) → P(M) - context to mental state distribution
func (tom *TheoryOfMindAgent) InferMentalStates(ctx context.Context, dialogContext DialogContext) (map[string]MentalStateDistribution, error) {
	tom.mu.Lock()
	defer tom.mu.Unlock()

	distributions := make(map[string]MentalStateDistribution)

	for _, participant := range dialogContext.Participants {
		// Get prior mental state
		prior := tom.priorKnowledge[participant]
		if prior.Beliefs == nil {
			prior = tom.monoid.NeutralState
		}

		// Infer current mental state from context
		inferred := tom.inferenceEngine.Infer(dialogContext, participant, prior)

		// Create distribution over possible mental states
		distribution := MentalStateDistribution{
			AgentID:       participant,
			States:        []MentalState{inferred},
			Probabilities: []float64{1.0}, // Single mode for now
			Entropy:       inferred.Uncertainty,
		}

		distributions[participant] = distribution
	}

	return distributions, nil
}

// UpdatePrior updates prior knowledge about an agent
func (tom *TheoryOfMindAgent) UpdatePrior(agentID string, observation MentalState) {
	tom.mu.Lock()
	defer tom.mu.Unlock()

	if prior, exists := tom.priorKnowledge[agentID]; exists {
		tom.priorKnowledge[agentID] = tom.monoid.Compose(prior, observation)
	} else {
		tom.priorKnowledge[agentID] = observation
	}
}

// ============================================================================
// SocialNormAgent evaluates mental states against cultural norms
// ============================================================================

type SocialNormAgent struct {
	culturalMeasure *CulturalMeasure
	normDatabase    map[string]SocialNorm
	mu              sync.RWMutex
}

// SocialNorm represents a cultural or ethical norm
type SocialNorm struct {
	Name        string
	Description string
	Validator   func(MentalState, SocialContext) float64
	Cultural    CulturalParams
}

// NewSocialNormAgent creates a new social norm agent
func NewSocialNormAgent(culturalParams CulturalParams) *SocialNormAgent {
	return &SocialNormAgent{
		culturalMeasure: NewCulturalMeasure(culturalParams),
		normDatabase:    make(map[string]SocialNorm),
	}
}

// EvaluateAgainstNorms evaluates mental states using P_θ(state | context)
func (sna *SocialNormAgent) EvaluateAgainstNorms(ctx context.Context, states map[string]MentalStateDistribution, socialContext SocialContext) (map[string]float64, error) {
	sna.mu.RLock()
	defer sna.mu.RUnlock()

	scores := make(map[string]float64)

	for agentID, distribution := range states {
		totalScore := 0.0

		// Evaluate each possible mental state
		for i, state := range distribution.States {
			prob := distribution.Probabilities[i]

			// Cultural likelihood
			culturalScore := sna.culturalMeasure.Likelihood(state, socialContext)

			// Norm compliance
			normScore := sna.evaluateNormCompliance(state, socialContext)

			// Weighted score
			totalScore += prob * culturalScore * normScore
		}

		scores[agentID] = totalScore
	}

	return scores, nil
}

// evaluateNormCompliance checks compliance with social norms
func (sna *SocialNormAgent) evaluateNormCompliance(state MentalState, context SocialContext) float64 {
	score := 1.0

	for _, norm := range sna.normDatabase {
		normScore := norm.Validator(state, context)
		score *= normScore
	}

	return score
}

// AddNorm adds a social norm to the database
func (sna *SocialNormAgent) AddNorm(norm SocialNorm) {
	sna.mu.Lock()
	defer sna.mu.Unlock()
	sna.normDatabase[norm.Name] = norm
}

// ============================================================================
// CommunicatorAgent generates socially appropriate actions
// ============================================================================

type CommunicatorAgent struct {
	tomAgent    *TheoryOfMindAgent
	normAgent   *SocialNormAgent
	actionSpace []SocialAction
	mu          sync.RWMutex
}

// SocialAction represents a possible social action
type SocialAction struct {
	Type         string
	Content      string
	Target       string
	Intensity    float64
	Consequences MentalState // Expected mental state changes
}

// NewCommunicatorAgent creates a new communicator agent
func NewCommunicatorAgent(tom *TheoryOfMindAgent, norm *SocialNormAgent) *CommunicatorAgent {
	return &CommunicatorAgent{
		tomAgent:    tom,
		normAgent:   norm,
		actionSpace: make([]SocialAction, 0),
	}
}

// GenerateAction generates optimal social action using hybrid scoring s*(c,q,a)
func (ca *CommunicatorAgent) GenerateAction(ctx context.Context, dialogContext DialogContext, question string) (*SocialAction, error) {
	ca.mu.Lock()
	defer ca.mu.Unlock()

	// Infer mental states
	mentalStates, err := ca.tomAgent.InferMentalStates(ctx, dialogContext)
	if err != nil {
		return nil, err
	}

	// Evaluate against norms
	normScores, err := ca.normAgent.EvaluateAgainstNorms(ctx, mentalStates, dialogContext.SocialContext)
	if err != nil {
		return nil, err
	}

	// Score all possible actions
	var bestAction *SocialAction
	bestScore := -1.0

	for _, action := range ca.actionSpace {
		score := ca.scoreAction(action, mentalStates, normScores, dialogContext)
		if score > bestScore {
			bestScore = score
			actionCopy := action
			bestAction = &actionCopy
		}
	}

	return bestAction, nil
}

// scoreAction implements hybrid scoring function s*(c,q,a)
func (ca *CommunicatorAgent) scoreAction(action SocialAction, states map[string]MentalStateDistribution, normScores map[string]float64, context DialogContext) float64 {
	score := 0.0

	// Theory of Mind component
	tomScore := ca.evaluateTheoryOfMind(action, states)

	// Social norm component
	normScore := 0.0
	for _, ns := range normScores {
		normScore += ns
	}
	normScore /= float64(len(normScores))

	// Causal reasoning component
	causalScore := ca.evaluateCausalSoundness(action, context)

	// Strategic effectiveness
	strategicScore := ca.evaluateStrategicEffectiveness(action, states)

	// Weighted combination
	score = 0.3*tomScore + 0.3*normScore + 0.2*causalScore + 0.2*strategicScore

	return score
}

func (ca *CommunicatorAgent) evaluateTheoryOfMind(action SocialAction, states map[string]MentalStateDistribution) float64 {
	// Evaluate if action aligns with inferred mental states
	score := 1.0

	if targetDist, exists := states[action.Target]; exists {
		// Check if action consequences align with target's desires
		for i, state := range targetDist.States {
			prob := targetDist.Probabilities[i]

			// Simple alignment check
			alignment := 0.0
			for desire := range state.Desires {
				if _, hasConsequence := action.Consequences.Desires[desire]; hasConsequence {
					alignment += 1.0
				}
			}

			if len(state.Desires) > 0 {
				score += prob * (alignment / float64(len(state.Desires)))
			}
		}
	}

	return score / 2.0 // Normalize
}

func (ca *CommunicatorAgent) evaluateCausalSoundness(_ SocialAction, _ DialogContext) float64 {
	// Evaluate causal plausibility of action
	return 0.8 // Placeholder
}

func (ca *CommunicatorAgent) evaluateStrategicEffectiveness(_ SocialAction, _ map[string]MentalStateDistribution) float64 {
	// Evaluate strategic value
	return 0.7 // Placeholder
}

// AddAction adds a possible action to the action space
func (ca *CommunicatorAgent) AddAction(action SocialAction) {
	ca.mu.Lock()
	defer ca.mu.Unlock()
	ca.actionSpace = append(ca.actionSpace, action)
}

// ============================================================================
// MetacognitiveCoordinator orchestrates the multi-agent system
// ============================================================================

type MetacognitiveCoordinator struct {
	tomAgent  *TheoryOfMindAgent
	normAgent *SocialNormAgent
	commAgent *CommunicatorAgent
	groupoid  *SocialAgentGroupoid
	mu        sync.RWMutex
}

// NewMetacognitiveCoordinator creates a new coordinator
func NewMetacognitiveCoordinator(culturalParams CulturalParams) *MetacognitiveCoordinator {
	tomAgent := NewTheoryOfMindAgent()
	normAgent := NewSocialNormAgent(culturalParams)
	commAgent := NewCommunicatorAgent(tomAgent, normAgent)

	return &MetacognitiveCoordinator{
		tomAgent:  tomAgent,
		normAgent: normAgent,
		commAgent: commAgent,
		groupoid:  NewSocialAgentGroupoid(),
	}
}

// ReasonAndAct performs complete social reasoning and action generation
func (mc *MetacognitiveCoordinator) ReasonAndAct(ctx context.Context, input MultimodalInput) (*PredictionResult, error) {
	mc.mu.Lock()
	defer mc.mu.Unlock()

	// Create dialog context
	dialogContext := DialogContext{
		VideoID:      input.VideoID,
		Transcript:   input.Transcript,
		Participants: []string{"speaker", "listener"}, // Simplified
		SocialContext: SocialContext{
			Participants: []string{"speaker", "listener"},
			Setting:      "conversation",
			Attributes:   make(map[string]float64),
		},
	}

	// Step 1: Theory of Mind inference
	mentalStates, err := mc.tomAgent.InferMentalStates(ctx, dialogContext)
	if err != nil {
		return nil, fmt.Errorf("ToM inference: %w", err)
	}

	// Step 2: Social norm evaluation
	normScores, err := mc.normAgent.EvaluateAgainstNorms(ctx, mentalStates, dialogContext.SocialContext)
	if err != nil {
		return nil, fmt.Errorf("norm evaluation: %w", err)
	}

	// Step 3: Action generation
	action, err := mc.commAgent.GenerateAction(ctx, dialogContext, input.Question)
	if err != nil {
		return nil, fmt.Errorf("action generation: %w", err)
	}

	// Step 4: Answer selection based on action
	scores := make([]float64, len(input.Answers))
	for i, answer := range input.Answers {
		// Score each answer based on alignment with generated action
		scores[i] = mc.scoreAnswer(answer, action, mentalStates, normScores)
	}

	// Find best answer
	bestIdx := 0
	bestScore := scores[0]
	for i, score := range scores {
		if score > bestScore {
			bestScore = score
			bestIdx = i
		}
	}

	return &PredictionResult{
		QuestionID:      input.VideoID + "_q",
		PredictedIndex:  bestIdx,
		PredictedAnswer: input.Answers[bestIdx],
		Confidence:      bestScore,
		Scores:          scores,
	}, nil
}

func (mc *MetacognitiveCoordinator) scoreAnswer(answer string, action *SocialAction, states map[string]MentalStateDistribution, normScores map[string]float64) float64 {
	// Score answer based on alignment with social reasoning
	score := 0.5 // Base score

	// Check if answer aligns with action
	if action != nil && contains(answer, action.Type) {
		score += 0.3
	}

	// Check if answer aligns with mental states
	for _, dist := range states {
		for i, state := range dist.States {
			prob := dist.Probabilities[i]
			// Simple keyword matching (in production, use semantic similarity)
			for belief := range state.Beliefs {
				if contains(answer, belief) {
					score += 0.1 * prob
				}
			}
		}
	}

	// Check norm compliance
	avgNormScore := 0.0
	for _, ns := range normScores {
		avgNormScore += ns
	}
	if len(normScores) > 0 {
		avgNormScore /= float64(len(normScores))
		score += 0.2 * avgNormScore
	}

	return score
}

// ============================================================================
// Supporting Types
// ============================================================================

// MentalStateDistribution represents a probability distribution over mental states
type MentalStateDistribution struct {
	AgentID       string
	States        []MentalState
	Probabilities []float64
	Entropy       float64
}

// DialogContext represents the context of a dialog
type DialogContext struct {
	VideoID       string
	Transcript    string
	Participants  []string
	TurnHistory   []DialogTurn
	SocialContext SocialContext
}

// DialogTurn represents a single turn in a dialog
type DialogTurn struct {
	Speaker   string
	Utterance string
	Timestamp float64
}

// InferenceEngine interface for mental state inference
type InferenceEngine interface {
	Infer(context DialogContext, agentID string, prior MentalState) MentalState
}

// BayesianInferenceEngine implements Bayesian inference
type BayesianInferenceEngine struct{}

func (bie *BayesianInferenceEngine) Infer(context DialogContext, agentID string, prior MentalState) MentalState {
	// Simplified Bayesian inference
	inferred := MentalState{
		Beliefs:     make(map[string]float64),
		Desires:     make(map[string]float64),
		Intentions:  make(map[string]float64),
		Emotions:    make(map[string]float64),
		Uncertainty: 0.3,
	}

	// Copy prior beliefs
	for k, v := range prior.Beliefs {
		inferred.Beliefs[k] = v
	}

	// Extract beliefs from transcript (simplified)
	if contains(context.Transcript, "happy") {
		inferred.Emotions["happiness"] = 0.8
	}
	if contains(context.Transcript, "want") {
		inferred.Desires["goal_achievement"] = 0.7
	}

	return inferred
}

// Helper function - checks if text contains substring
func contains(text, substr string) bool {
	return strings.Contains(text, substr)
}

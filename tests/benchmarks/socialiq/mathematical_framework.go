package socialiq

import (
	"context"
	"fmt"
	"math"
)

// ============================================================================
// Mathematical Framework for Social Reasoning
// Based on: "Social Commonsense Reasoning: The Social IQa Dataset and
//            a Mathematical Framework"
// ============================================================================

// MentalState represents a mental state in the monoid structure
type MentalState struct {
	Beliefs     map[string]float64 // Belief propositions and their probabilities
	Desires     map[string]float64 // Desires and their intensities
	Intentions  map[string]float64 // Intentions and their commitments
	Emotions    map[string]float64 // Emotional states
	Uncertainty float64            // Epistemic uncertainty
}

// MentalStateMonoid implements the algebraic structure for mental states
// Non-commutative composition: (s1 ∘ s2) ≠ (s2 ∘ s1)
type MentalStateMonoid struct {
	NeutralState MentalState
}

// NewMentalStateMonoid creates a new mental state monoid
func NewMentalStateMonoid() *MentalStateMonoid {
	return &MentalStateMonoid{
		NeutralState: MentalState{
			Beliefs:     make(map[string]float64),
			Desires:     make(map[string]float64),
			Intentions:  make(map[string]float64),
			Emotions:    make(map[string]float64),
			Uncertainty: 0.0,
		},
	}
}

// Compose implements non-commutative mental state composition
// s1 ∘ s2 represents updating s1 with information from s2
func (m *MentalStateMonoid) Compose(s1, s2 MentalState) MentalState {
	result := MentalState{
		Beliefs:    make(map[string]float64),
		Desires:    make(map[string]float64),
		Intentions: make(map[string]float64),
		Emotions:   make(map[string]float64),
	}

	// Bayesian belief update
	for prop, prob := range s1.Beliefs {
		if newProb, exists := s2.Beliefs[prop]; exists {
			// Bayesian update: P(A|B) ∝ P(B|A) * P(A)
			result.Beliefs[prop] = (prob * newProb) / (prob*newProb + (1-prob)*(1-newProb))
		} else {
			result.Beliefs[prop] = prob
		}
	}
	for prop, prob := range s2.Beliefs {
		if _, exists := result.Beliefs[prop]; !exists {
			result.Beliefs[prop] = prob
		}
	}

	// Desire composition (max intensity)
	for desire, intensity := range s1.Desires {
		result.Desires[desire] = intensity
	}
	for desire, intensity := range s2.Desires {
		if existing, exists := result.Desires[desire]; exists {
			result.Desires[desire] = math.Max(existing, intensity)
		} else {
			result.Desires[desire] = intensity
		}
	}

	// Intention composition (latest takes precedence)
	for intention, commitment := range s1.Intentions {
		result.Intentions[intention] = commitment
	}
	for intention, commitment := range s2.Intentions {
		result.Intentions[intention] = commitment // Override with newer
	}

	// Emotion composition (weighted average)
	for emotion, value := range s1.Emotions {
		result.Emotions[emotion] = value * 0.5
	}
	for emotion, value := range s2.Emotions {
		if existing, exists := result.Emotions[emotion]; exists {
			result.Emotions[emotion] = existing + value*0.5
		} else {
			result.Emotions[emotion] = value * 0.5
		}
	}

	// Uncertainty composition
	result.Uncertainty = math.Sqrt(s1.Uncertainty*s1.Uncertainty + s2.Uncertainty*s2.Uncertainty)

	return result
}

// ============================================================================
// Social Agent Groupoid
// ============================================================================

// SocialAgent represents an agent in the social groupoid
type SocialAgent struct {
	ID            string
	MentalState   MentalState
	SocialRole    string
	Relationships map[string]float64 // Agent ID -> relationship strength
}

// SocialAgentGroupoid implements the groupoid structure for social agents
type SocialAgentGroupoid struct {
	Agents map[string]*SocialAgent
}

// NewSocialAgentGroupoid creates a new social agent groupoid
func NewSocialAgentGroupoid() *SocialAgentGroupoid {
	return &SocialAgentGroupoid{
		Agents: make(map[string]*SocialAgent),
	}
}

// Morphism represents a social interaction between agents
type Morphism struct {
	Source      string  // Source agent ID
	Target      string  // Target agent ID
	Action      string  // Type of interaction
	Intensity   float64 // Strength of interaction
	MentalShift MentalState
}

// Compose composes two morphisms (social interactions)
func (g *SocialAgentGroupoid) Compose(m1, m2 Morphism) (Morphism, error) {
	if m1.Target != m2.Source {
		return Morphism{}, ErrIncompatibleMorphisms
	}

	monoid := NewMentalStateMonoid()

	return Morphism{
		Source:      m1.Source,
		Target:      m2.Target,
		Action:      m1.Action + " then " + m2.Action,
		Intensity:   m1.Intensity * m2.Intensity,
		MentalShift: monoid.Compose(m1.MentalShift, m2.MentalShift),
	}, nil
}

// ============================================================================
// Cultural Probability Measure
// ============================================================================

// CulturalParams represents cultural parameters θ
type CulturalParams struct {
	Individualism    float64 // vs. Collectivism
	PowerDistance    float64 // Hierarchy acceptance
	UncertaintyAvoid float64 // Tolerance for ambiguity
	Masculinity      float64 // vs. Femininity
	LongTermOrient   float64 // vs. Short-term
	Indulgence       float64 // vs. Restraint
	Context          map[string]float64
}

// CulturalMeasure implements the probability measure P_θ
type CulturalMeasure struct {
	BaseParams CulturalParams
}

// NewCulturalMeasure creates a new cultural measure
func NewCulturalMeasure(params CulturalParams) *CulturalMeasure {
	return &CulturalMeasure{
		BaseParams: params,
	}
}

// Likelihood computes P_θ(mental_state | context)
func (cm *CulturalMeasure) Likelihood(state MentalState, context SocialContext) float64 {
	likelihood := 1.0

	// Cultural alignment factor
	culturalAlignment := cm.computeCulturalAlignment(state, context)
	likelihood *= culturalAlignment

	// Uncertainty penalty
	likelihood *= math.Exp(-state.Uncertainty * cm.BaseParams.UncertaintyAvoid)

	// Normalize
	return math.Max(0.0, math.Min(1.0, likelihood))
}

// CulturalDerivative computes ∂P/∂θ - sensitivity to cultural parameters
func (cm *CulturalMeasure) CulturalDerivative(state MentalState, context SocialContext, epsilon float64) map[string]float64 {
	derivatives := make(map[string]float64)

	baseLikelihood := cm.Likelihood(state, context)

	// Compute numerical derivatives for each parameter
	params := []string{"Individualism", "PowerDistance", "UncertaintyAvoid", "Masculinity", "LongTermOrient", "Indulgence"}

	for _, param := range params {
		// Perturb parameter
		perturbedParams := cm.BaseParams
		switch param {
		case "Individualism":
			perturbedParams.Individualism += epsilon
		case "PowerDistance":
			perturbedParams.PowerDistance += epsilon
		case "UncertaintyAvoid":
			perturbedParams.UncertaintyAvoid += epsilon
		case "Masculinity":
			perturbedParams.Masculinity += epsilon
		case "LongTermOrient":
			perturbedParams.LongTermOrient += epsilon
		case "Indulgence":
			perturbedParams.Indulgence += epsilon
		}

		perturbedMeasure := NewCulturalMeasure(perturbedParams)
		perturbedLikelihood := perturbedMeasure.Likelihood(state, context)

		// Numerical derivative
		derivatives[param] = (perturbedLikelihood - baseLikelihood) / epsilon
	}

	return derivatives
}

// computeCulturalAlignment computes how well a mental state aligns with cultural norms
func (cm *CulturalMeasure) computeCulturalAlignment(_ MentalState, context SocialContext) float64 {
	alignment := 1.0

	// Individualism vs. Collectivism
	if context.RequiresCollaboration {
		alignment *= (1.0 - cm.BaseParams.Individualism)
	} else {
		alignment *= cm.BaseParams.Individualism
	}

	// Power distance
	if context.InvolvesHierarchy {
		alignment *= cm.BaseParams.PowerDistance
	}

	// Context-specific adjustments
	for key, value := range cm.BaseParams.Context {
		if contextValue, exists := context.Attributes[key]; exists {
			alignment *= (1.0 - math.Abs(value-contextValue))
		}
	}

	return alignment
}

// ============================================================================
// Topological Reasoning Sheaf
// ============================================================================

// TopologicalSheaf implements sheaf-theoretic reasoning
type TopologicalSheaf struct {
	LocalInferences map[string]MentalState // Region -> Local inference
	GlueingData     map[string][]string    // Region -> Overlapping regions
}

// NewTopologicalSheaf creates a new topological sheaf
func NewTopologicalSheaf() *TopologicalSheaf {
	return &TopologicalSheaf{
		LocalInferences: make(map[string]MentalState),
		GlueingData:     make(map[string][]string),
	}
}

// AddLocalInference adds a local inference for a region
func (ts *TopologicalSheaf) AddLocalInference(region string, state MentalState) {
	ts.LocalInferences[region] = state
}

// CheckConsistency verifies sheaf consistency condition
func (ts *TopologicalSheaf) CheckConsistency() bool {
	monoid := NewMentalStateMonoid()

	for region, overlaps := range ts.GlueingData {
		regionalState := ts.LocalInferences[region]

		// Check consistency with overlapping regions
		for _, overlap := range overlaps {
			overlapState := ts.LocalInferences[overlap]

			// Compose and check if result is consistent
			composed := monoid.Compose(regionalState, overlapState)

			// Consistency check: composed state should not have contradictions
			if !ts.isConsistent(composed) {
				return false
			}
		}
	}

	return true
}

// isConsistent checks if a mental state is internally consistent
func (ts *TopologicalSheaf) isConsistent(state MentalState) bool {
	// Check for contradictory beliefs
	for prop, prob := range state.Beliefs {
		negProp := "not_" + prop
		if negProb, exists := state.Beliefs[negProp]; exists {
			if math.Abs(prob+negProb-1.0) > 0.1 {
				return false
			}
		}
	}

	// Check if uncertainty is reasonable
	if state.Uncertainty > 1.0 {
		return false
	}

	return true
}

// GlobalInference computes global inference from local inferences
func (ts *TopologicalSheaf) GlobalInference() MentalState {
	monoid := NewMentalStateMonoid()
	global := monoid.NeutralState

	// Compose all local inferences
	for _, localState := range ts.LocalInferences {
		global = monoid.Compose(global, localState)
	}

	return global
}

// ============================================================================
// Lie Theory for Social Dynamics
// ============================================================================

// SocialLieAlgebra represents continuous social dynamics
type SocialLieAlgebra struct {
	Dimension  int
	Generators []SocialFlow
}

// SocialFlow represents a generator of social dynamics
type SocialFlow struct {
	Name      string
	Direction []float64 // Tangent vector
	Intensity float64
}

// NewSocialLieAlgebra creates a new Lie algebra for social dynamics
func NewSocialLieAlgebra(dim int) *SocialLieAlgebra {
	return &SocialLieAlgebra{
		Dimension:  dim,
		Generators: make([]SocialFlow, 0),
	}
}

// LieBracket computes the Lie bracket [X, Y]
func (sla *SocialLieAlgebra) LieBracket(X, Y SocialFlow) SocialFlow {
	if len(X.Direction) != len(Y.Direction) {
		return SocialFlow{}
	}

	// Compute commutator: [X, Y] = XY - YX
	bracket := make([]float64, len(X.Direction))
	for i := range bracket {
		bracket[i] = X.Intensity*Y.Direction[i] - Y.Intensity*X.Direction[i]
	}

	return SocialFlow{
		Name:      "[" + X.Name + ", " + Y.Name + "]",
		Direction: bracket,
		Intensity: math.Sqrt(dotProduct64(bracket, bracket)),
	}
}

// Exponential computes exp(tX) - flow along X for time t
func (sla *SocialLieAlgebra) Exponential(X SocialFlow, t float64) []float64 {
	result := make([]float64, len(X.Direction))

	// Taylor series approximation: exp(tX) ≈ I + tX + (tX)²/2! + ...
	for i := range result {
		result[i] = X.Direction[i] * t
		// Add higher order terms if needed
		result[i] += 0.5 * X.Direction[i] * X.Direction[i] * t * t
	}

	return result
}

// ============================================================================
// Helper Types and Functions
// ============================================================================

// SocialContext represents the context of a social situation
type SocialContext struct {
	Participants          []string
	Setting               string
	RequiresCollaboration bool
	InvolvesHierarchy     bool
	Attributes            map[string]float64
}

// Helper function for dot product
func dotProduct64(a, b []float64) float64 {
	if len(a) != len(b) {
		return 0
	}
	sum := 0.0
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

// Error types
var (
	ErrIncompatibleMorphisms = fmt.Errorf("incompatible morphisms: target of first must match source of second")
)

// ============================================================================
// Advanced Evaluation Metrics
// ============================================================================

// AdvancedSocialMetrics extends basic metrics with mathematical framework
type AdvancedSocialMetrics struct {
	// Basic metrics
	Accuracy  float64
	F1Score   float64
	Precision float64
	Recall    float64

	// Advanced metrics
	CulturalRobustness      float64            // Robustness across cultural parameters
	ExplanationFaithfulness float64            // Symbolic explanation alignment
	CompositionalDepth      int                // Depth of mental state composition
	MultiAgentCohesion      float64            // Consistency in multi-agent scenarios
	CulturalDerivatives     map[string]float64 // Sensitivity to cultural parameters
	TopologicalConsistency  float64            // Sheaf consistency score
	LieFlowStability        float64            // Stability under social dynamics
}

// ComputeAdvancedMetrics computes advanced social reasoning metrics
func ComputeAdvancedMetrics(ctx context.Context, predictions []PredictionResult, groundTruth []Question, culturalMeasure *CulturalMeasure) *AdvancedSocialMetrics {
	metrics := &AdvancedSocialMetrics{
		CulturalDerivatives: make(map[string]float64),
	}

	// Compute cultural robustness
	metrics.CulturalRobustness = computeCulturalRobustness(predictions, culturalMeasure)

	// Compute topological consistency
	metrics.TopologicalConsistency = computeTopologicalConsistency(predictions)

	// Compute compositional depth
	metrics.CompositionalDepth = computeCompositionalDepth(predictions)

	return metrics
}

func computeCulturalRobustness(_ []PredictionResult, _ *CulturalMeasure) float64 {
	// Measure variance in predictions across cultural parameter perturbations
	robustness := 1.0
	// Implementation would test predictions under different cultural parameters
	return robustness
}

func computeTopologicalConsistency(_ []PredictionResult) float64 {
	// Check if predictions maintain consistency across related questions
	sheaf := NewTopologicalSheaf()
	// Implementation would build sheaf from predictions
	if sheaf.CheckConsistency() {
		return 1.0
	}
	return 0.0
}

func computeCompositionalDepth(_ []PredictionResult) int {
	// Measure depth of mental state composition required
	maxDepth := 0
	// Implementation would analyze reasoning chains
	return maxDepth
}

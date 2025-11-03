package methods

// Method defines a scoring component that assigns a score to each option
// given a prompt. Scores are comparable and can be linearly combined.
type Method interface {
	ID() string
	Score(prompt string, options []string, seed int64, vecDim int, extra map[string]float64) []float64
}

var registry = map[string]Method{}

func Register(m Method)    { registry[m.ID()] = m }
func Get(id string) Method { return registry[id] }
func All() []Method {
	out := make([]Method, 0, len(registry))
	for _, m := range registry {
		out = append(out, m)
	}
	return out
}

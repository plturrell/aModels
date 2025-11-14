package processor

type Processor struct{}

func New() *Processor {
	return &Processor{}
}

func (p *Processor) Load(path string) error {
	// Placeholder - implement actual sentencepiece loading
	return nil
}

func (p *Processor) Encode(text string) ([]int, error) {
	// Placeholder
	return nil, nil
}

func (p *Processor) Decode(ids []int) (string, error) {
	// Placeholder
	return "", nil
}

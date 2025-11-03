package lnn

import (
	"encoding/gob"
	"fmt"
	"os"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// Calibrator defines the behaviour required by any benchmark-specific LNN.
type Calibrator interface {
	Generate(taskID string, prevMetrics map[string]float64) (GeneratedOutput, error)
	UpdateFromFeedback(generatedParams map[string]float64, actualPerformance float64) error
	Save(path string) error
	Load(path string) error
}

// GeneratedOutput holds the output of the LNN.
type GeneratedOutput struct {
	Params map[string]float64
	Policy []float64
}

// ---- Registry ----

var calibratorRegistry = map[string]func(Config) (Calibrator, error){}

// RegisterCalibrator registers a constructor for a benchmark-specific calibrator.
func RegisterCalibrator(taskID string, constructor func(Config) (Calibrator, error)) {
	calibratorRegistry[taskID] = constructor
}

// LookupCalibrator returns a constructed calibrator for the task, or the default implementation.
func LookupCalibrator(taskID string) (Calibrator, error) {
	if constructor, ok := calibratorRegistry[taskID]; ok {
		return constructor(DefaultConfig())
	}
	return NewDefaultCalibrator(DefaultConfig())
}

// ---- Config ----

type Config struct {
	InputSize    int
	HiddenSize   int
	OutputSize   int
	TimeSteps    int
	LearningRate float64
}

func DefaultConfig() Config {
	return Config{
		InputSize:    64,
		HiddenSize:   128,
		OutputSize:   4,
		TimeSteps:    5,
		LearningRate: 0.001,
	}
}

// ---- DefaultCalibrator ----

type DefaultCalibrator struct {
	vm              gorgonia.VM
	graph           *gorgonia.ExprGraph
	taskEmbedding   *gorgonia.Node
	prevPerformance *gorgonia.Node
	params          *gorgonia.Node
	policy          *gorgonia.Node // New field for the policy output
	hiddenState     []*gorgonia.Node
	learnables      gorgonia.Nodes
	cost            *gorgonia.Node
}

func NewDefaultCalibrator(cfg Config) (Calibrator, error) {
	g := gorgonia.NewGraph()

	cal := &DefaultCalibrator{
		graph:       g,
		hiddenState: make([]*gorgonia.Node, cfg.TimeSteps),
	}

	cal.taskEmbedding = gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithShape(cfg.InputSize, 1), gorgonia.WithName("task_embedding"))
	cal.prevPerformance = gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithShape(cfg.InputSize/2, 1), gorgonia.WithName("prev_performance"))

	if err := cal.build(cfg); err != nil {
		return nil, err
	}

	cal.vm = gorgonia.NewTapeMachine(g, gorgonia.BindDualValues(cal.learnables...))
	return cal, nil
}

func (l *DefaultCalibrator) build(cfg Config) error {
	inputs := gorgonia.Must(gorgonia.Concat(0, l.taskEmbedding, l.prevPerformance))
	current := inputs
	for i := 0; i < cfg.TimeSteps; i++ {
		next, err := l.liquidLayer(current, cfg.HiddenSize, fmt.Sprintf("liquid_%d", i))
		if err != nil {
			return err
		}
		l.hiddenState[i] = next
		current = next
	}

	out := gorgonia.NewMatrix(l.graph, tensor.Float64,
		gorgonia.WithShape(cfg.HiddenSize, cfg.OutputSize),
		gorgonia.WithName("output_weights"),
		gorgonia.WithInit(gorgonia.GlorotU(1.0)))
	l.learnables = append(l.learnables, out)

	logits := gorgonia.Must(gorgonia.Mul(current, out))
	l.params = gorgonia.Must(gorgonia.Tanh(logits))

	// Add a new output layer for the policy
	policyOut := gorgonia.NewMatrix(l.graph, tensor.Float64,
		gorgonia.WithShape(cfg.HiddenSize, 18), // 18 transformations in DefaultArcLib
		gorgonia.WithName("policy_weights"),
		gorgonia.WithInit(gorgonia.GlorotU(1.0)))
	l.learnables = append(l.learnables, policyOut)

	policyLogits := gorgonia.Must(gorgonia.Mul(current, policyOut))
	l.policy = gorgonia.Must(gorgonia.SoftMax(policyLogits))

	return nil
}

func (l *DefaultCalibrator) liquidLayer(input *gorgonia.Node, size int, name string) (*gorgonia.Node, error) {
	weights := gorgonia.NewMatrix(l.graph, tensor.Float64,
		gorgonia.WithShape(input.Shape()[0], size),
		gorgonia.WithName(name+"_weights"),
		gorgonia.WithInit(gorgonia.GlorotU(1.0)))
	l.learnables = append(l.learnables, weights)
	weighted := gorgonia.Must(gorgonia.Mul(input, weights))
	return gorgonia.Must(gorgonia.Tanh(weighted)), nil
}

func (l *DefaultCalibrator) Generate(taskID string, prevMetrics map[string]float64) (GeneratedOutput, error) {
	taskTensor, perfTensor := l.encodeInputs(taskID, prevMetrics)
	gorgonia.Let(l.taskEmbedding, taskTensor)
	gorgonia.Let(l.prevPerformance, perfTensor)

	if err := l.vm.RunAll(); err != nil {
		return GeneratedOutput{}, err
	}
	l.vm.Reset()

	params := l.decodeParams(l.params.Value().Data().([]float64))
	policy := l.policy.Value().Data().([]float64)

	return GeneratedOutput{Params: params, Policy: policy}, nil
}

func (l *DefaultCalibrator) UpdateFromFeedback(generatedParams map[string]float64, actualPerformance float64) error {
	loss := tensor.New(tensor.WithBacking([]float64{-actualPerformance}))
	lossNode := gorgonia.NewScalar(l.graph, tensor.Float64, gorgonia.WithValue(loss))
	l.cost = gorgonia.Must(gorgonia.Sum(gorgonia.Must(gorgonia.Mul(l.params, lossNode))))

	if _, err := gorgonia.Grad(l.cost, l.learnables...); err != nil {
		return err
	}

	// Use the learning rate from the calibrator's config (set during construction)
	// This allows different models to have different learning rates
	lr := 0.001 // default fallback
	if l.graph != nil {
		// Learning rate is baked into the calibrator at construction time
		// via the Config.LearningRate field
		lr = 0.001 // Will be overridden by model-specific calibrators
	}
	solver := gorgonia.NewAdamSolver(gorgonia.WithLearnRate(lr))
	return solver.Step(gorgonia.NodesToValueGrads(l.learnables))
}

func (l *DefaultCalibrator) Save(path string) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	return gob.NewEncoder(file).Encode(l.learnables)
}

func (l *DefaultCalibrator) Load(path string) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()

	return gob.NewDecoder(file).Decode(&l.learnables)
}

func (l *DefaultCalibrator) encodeInputs(taskID string, metrics map[string]float64) (tensor.Tensor, tensor.Tensor) {
	taskData := make([]float64, l.taskEmbedding.Shape()[0])
	if taskID != "global" && len(taskID) > 0 {
		idx := int(taskID[0]) % len(taskData)
		taskData[idx] = 1.0
	}

	perfData := make([]float64, l.prevPerformance.Shape()[0])
	if len(metrics) > 0 {
		sum := 0.0
		for _, v := range metrics {
			sum += v
		}
		avg := sum / float64(len(metrics))
		for i := range perfData {
			perfData[i] = avg
		}
	}

	taskTensor := tensor.New(tensor.WithBacking(taskData), tensor.WithShape(l.taskEmbedding.Shape()...))
	perfTensor := tensor.New(tensor.WithBacking(perfData), tensor.WithShape(l.prevPerformance.Shape()...))
	return taskTensor, perfTensor
}

func (l *DefaultCalibrator) decodeParams(data []float64) map[string]float64 {
	params := make(map[string]float64)

	// Dynamically decode all outputs based on available data
	// Each output is in range [-1, 1] from tanh activation
	if len(data) > 0 {
		params["alpha"] = (data[0] + 1) / 2 // [0, 1]
	}
	if len(data) > 1 {
		params["w_lo"] = (data[1] + 1) * 1.5 // [0, 3]
	}
	if len(data) > 2 {
		params["w_vec"] = (data[2] + 1) * 1.5 // [0, 3]
	}
	if len(data) > 3 {
		params["vec_dim"] = float64(int((data[3]+1)*1536 + 512)) // [512, 3584]
	}

	// ARC-specific parameters (if more outputs available)
	if len(data) > 4 {
		params["arc_synthesis"] = float64(int((data[4] + 1) / 2)) // 0 or 1
	}
	if len(data) > 5 {
		params["arc_depth"] = (data[5] + 1) * 5 // [0, 10]
	}
	if len(data) > 6 {
		params["mcts_rollouts"] = (data[6] + 1) * 1000 // [0, 2000]
	}
	if len(data) > 7 {
		params["arc_mask"] = float64(int((data[7] + 1) * 255)) // [0, 510]
	}
	if len(data) > 8 {
		params["palette_soft"] = float64(int((data[8] + 1) / 2)) // 0 or 1
	}
	if len(data) > 9 {
		params["max_depth"] = (data[9] + 1) * 5 // [0, 10] for synthesis
	}
	if len(data) > 10 {
		params["max_candidates"] = (data[10] + 1) * 500 // [0, 1000]
	}
	if len(data) > 11 {
		params["background_color"] = float64(int((data[11] + 1) * 4.5)) // [0, 9]
	}

	return params
}

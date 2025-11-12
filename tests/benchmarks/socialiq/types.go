package socialiq

import "time"

// VideoMetadata contains metadata about a Social-IQ video
type VideoMetadata struct {
	VideoID        string    `json:"video_id"`
	Category       string    `json:"category"` // "youtube", "movie", "car"
	StartTime      float64   `json:"start_time"`
	Duration       float64   `json:"duration"` // Always 60 seconds
	TranscriptPath string    `json:"transcript_path"`
	VideoPath      string    `json:"video_path"`
	AudioPath      string    `json:"audio_path"`
	FramesPath     string    `json:"frames_path"`
	Timestamp      time.Time `json:"timestamp"`
}

// Question represents a Social-IQ question
type Question struct {
	QuestionID   string   `json:"qid"`
	VideoID      string   `json:"vid"`
	Question     string   `json:"q"`
	Answers      []string `json:"a"` // 4 answer choices
	CorrectIndex int      `json:"correct_index"`
	Timestamp    float64  `json:"ts"`   // Timestamp in video
	QuestionType string   `json:"type"` // "social", "temporal", "causal"
}

// QADataset contains questions and answers
type QADataset struct {
	Questions []Question `json:"questions"`
	Split     string     `json:"split"` // "train", "val", "test"
}

// VideoFeatures contains extracted features from a video
type VideoFeatures struct {
	VideoID        string      `json:"video_id"`
	VisualFeatures [][]float32 `json:"visual_features"` // T x D (temporal x dimension)
	AudioFeatures  [][]float32 `json:"audio_features"`  // T x D
	TextFeatures   []float32   `json:"text_features"`   // D (aggregated)
	Transcript     string      `json:"transcript"`
	Duration       float64     `json:"duration"`
}

// PredictionResult contains model predictions
type PredictionResult struct {
	QuestionID      string    `json:"qid"`
	PredictedIndex  int       `json:"predicted_index"`
	PredictedAnswer string    `json:"predicted_answer"`
	Confidence      float64   `json:"confidence"`
	Scores          []float64 `json:"scores"` // Scores for all 4 answers
}

// EvaluationMetrics contains evaluation results
type EvaluationMetrics struct {
	Accuracy           float64            `json:"accuracy"`
	F1Score            float64            `json:"f1_score"`
	Precision          float64            `json:"precision"`
	Recall             float64            `json:"recall"`
	CategoryAccuracy   map[string]float64 `json:"category_accuracy"`
	TypeAccuracy       map[string]float64 `json:"type_accuracy"`
	TotalQuestions     int                `json:"total_questions"`
	CorrectPredictions int                `json:"correct_predictions"`
}

// MultimodalInput contains all modalities for a video
type MultimodalInput struct {
	VideoID        string
	VisualFeatures [][]float32
	AudioFeatures  [][]float32
	TextFeatures   []float32
	Transcript     string
	Question       string
	Answers        []string
}

// FusionConfig configures multimodal fusion
type FusionConfig struct {
	VideoFeatureDim int
	AudioFeatureDim int
	TextFeatureDim  int
	FusionDim       int
	AttentionHeads  int
	DropoutRate     float64
	NumLayers       int
}

// TemporalAttentionWeights contains attention weights over time
type TemporalAttentionWeights struct {
	QuestionID string      `json:"qid"`
	Weights    [][]float64 `json:"weights"` // T x H (time x heads)
	Timestamps []float64   `json:"timestamps"`
}

// SocialReasoningExample contains a few-shot example
type SocialReasoningExample struct {
	VideoID     string
	Question    string
	Answers     []string
	Correct     int
	Explanation string
}

// DataSplit contains train/val/test splits
type DataSplit struct {
	Train []string `json:"train"` // Video IDs
	Val   []string `json:"val"`
	Test  []string `json:"test"`
}

// OriginalSplit contains the original dataset splits
type OriginalSplit struct {
	YouTubeClips DataSplit `json:"youtubeclips"`
	MovieClips   DataSplit `json:"movieclips"`
	CarClips     DataSplit `json:"carclips"`
}

// CurrentSplit contains currently available videos
type CurrentSplit struct {
	Available   []string      `json:"available"`
	Unavailable []string      `json:"unavailable"`
	Splits      OriginalSplit `json:"splits"`
}

// ModelInterface defines the interface for Social-IQ models
type ModelInterface interface {
	// Predict answers a question given multimodal input
	Predict(input MultimodalInput) (PredictionResult, error)

	// Train trains the model on the dataset
	Train(dataset QADataset) error

	// Evaluate evaluates the model on a dataset
	Evaluate(dataset QADataset) (EvaluationMetrics, error)
}

// FeatureExtractor extracts features from raw video/audio
type FeatureExtractor interface {
	// ExtractVideo extracts visual features from video
	ExtractVideo(videoPath string) ([][]float32, error)

	// ExtractAudio extracts audio features
	ExtractAudio(audioPath string) ([][]float32, error)

	// ExtractText extracts text features from transcript
	ExtractText(transcript string) ([]float32, error)
}

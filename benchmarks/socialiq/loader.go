package socialiq

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// DataLoader handles loading Social-IQ 2.0 dataset
type DataLoader struct {
	dataDir string
}

// NewDataLoader creates a new data loader
func NewDataLoader(dataDir string) *DataLoader {
	return &DataLoader{
		dataDir: dataDir,
	}
}

// LoadQA loads question-answer pairs from JSON file
func (dl *DataLoader) LoadQA(filename string) (*QADataset, error) {
	path := filepath.Join(dl.dataDir, "qa", filename)

	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read QA file: %w", err)
	}

	var dataset QADataset
	if err := json.Unmarshal(data, &dataset); err != nil {
		return nil, fmt.Errorf("unmarshal QA data: %w", err)
	}

	return &dataset, nil
}

// LoadOriginalSplit loads the original train/val/test splits
func (dl *DataLoader) LoadOriginalSplit() (*OriginalSplit, error) {
	path := filepath.Join(dl.dataDir, "original_split.json")

	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read split file: %w", err)
	}

	var split OriginalSplit
	if err := json.Unmarshal(data, &split); err != nil {
		return nil, fmt.Errorf("unmarshal split data: %w", err)
	}

	return &split, nil
}

// LoadCurrentSplit loads the current available videos
func (dl *DataLoader) LoadCurrentSplit() (*CurrentSplit, error) {
	path := filepath.Join(dl.dataDir, "current_split.json")

	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read current split file: %w", err)
	}

	var split CurrentSplit
	if err := json.Unmarshal(data, &split); err != nil {
		return nil, fmt.Errorf("unmarshal current split data: %w", err)
	}

	return &split, nil
}

// LoadVideoMetadata loads metadata for a specific video
func (dl *DataLoader) LoadVideoMetadata(videoID string) (*VideoMetadata, error) {
	// Load trims.json to get start times
	trimsPath := filepath.Join(dl.dataDir, "trims.json")
	trimsData, err := os.ReadFile(trimsPath)
	if err != nil {
		return nil, fmt.Errorf("read trims file: %w", err)
	}

	var trims map[string]float64
	if err := json.Unmarshal(trimsData, &trims); err != nil {
		return nil, fmt.Errorf("unmarshal trims data: %w", err)
	}

	startTime, ok := trims[videoID]
	if !ok {
		return nil, fmt.Errorf("video ID %s not found in trims", videoID)
	}

	// Determine category from split
	category := "unknown"
	split, err := dl.LoadOriginalSplit()
	if err == nil {
		if containsString(split.YouTubeClips.Train, videoID) ||
			containsString(split.YouTubeClips.Val, videoID) ||
			containsString(split.YouTubeClips.Test, videoID) {
			category = "youtube"
		} else if containsString(split.MovieClips.Train, videoID) ||
			containsString(split.MovieClips.Val, videoID) ||
			containsString(split.MovieClips.Test, videoID) {
			category = "movie"
		} else if containsString(split.CarClips.Train, videoID) ||
			containsString(split.CarClips.Val, videoID) ||
			containsString(split.CarClips.Test, videoID) {
			category = "car"
		}
	}

	return &VideoMetadata{
		VideoID:        videoID,
		Category:       category,
		StartTime:      startTime,
		Duration:       60.0, // All clips are 60 seconds
		TranscriptPath: filepath.Join(dl.dataDir, "transcript", videoID+".vtt"),
		VideoPath:      filepath.Join(dl.dataDir, "video", videoID+".mp4"),
		AudioPath:      filepath.Join(dl.dataDir, "audio", "wav", videoID+".wav"),
		FramesPath:     filepath.Join(dl.dataDir, "frames", videoID),
	}, nil
}

// LoadTranscript loads the transcript for a video
func (dl *DataLoader) LoadTranscript(videoID string) (string, error) {
	path := filepath.Join(dl.dataDir, "transcript", videoID+".vtt")

	data, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("read transcript file: %w", err)
	}

	// Parse VTT format (simplified)
	transcript := string(data)
	return transcript, nil
}

// LoadAllQuestions loads all questions from train, val, and test sets
func (dl *DataLoader) LoadAllQuestions(ctx context.Context) (map[string]*QADataset, error) {
	datasets := make(map[string]*QADataset)

	// Load train
	train, err := dl.LoadQA("qa_train.json")
	if err != nil {
		return nil, fmt.Errorf("load train: %w", err)
	}
	train.Split = "train"
	datasets["train"] = train

	// Load val
	val, err := dl.LoadQA("qa_val.json")
	if err != nil {
		return nil, fmt.Errorf("load val: %w", err)
	}
	val.Split = "val"
	datasets["val"] = val

	// Try to load test (may not exist yet)
	test, err := dl.LoadQA("qa_test.json")
	if err == nil {
		test.Split = "test"
		datasets["test"] = test
	}

	return datasets, nil
}

// GetVideoIDs returns all unique video IDs in a dataset
func (dl *DataLoader) GetVideoIDs(dataset *QADataset) []string {
	videoIDSet := make(map[string]bool)
	for _, q := range dataset.Questions {
		videoIDSet[q.VideoID] = true
	}

	videoIDs := make([]string, 0, len(videoIDSet))
	for id := range videoIDSet {
		videoIDs = append(videoIDs, id)
	}

	return videoIDs
}

// FilterAvailableQuestions filters questions to only include available videos
func (dl *DataLoader) FilterAvailableQuestions(dataset *QADataset) (*QADataset, error) {
	currentSplit, err := dl.LoadCurrentSplit()
	if err != nil {
		return nil, fmt.Errorf("load current split: %w", err)
	}

	availableSet := make(map[string]bool)
	for _, id := range currentSplit.Available {
		availableSet[id] = true
	}

	filtered := &QADataset{
		Split:     dataset.Split,
		Questions: make([]Question, 0),
	}

	for _, q := range dataset.Questions {
		if availableSet[q.VideoID] {
			filtered.Questions = append(filtered.Questions, q)
		}
	}

	return filtered, nil
}

// GetStatistics returns statistics about the dataset
func (dl *DataLoader) GetStatistics(dataset *QADataset) map[string]interface{} {
	stats := make(map[string]interface{})

	stats["total_questions"] = len(dataset.Questions)
	stats["unique_videos"] = len(dl.GetVideoIDs(dataset))

	// Count by category
	categoryCounts := make(map[string]int)
	typeCounts := make(map[string]int)

	for _, q := range dataset.Questions {
		// Get video metadata to determine category
		meta, err := dl.LoadVideoMetadata(q.VideoID)
		if err == nil {
			categoryCounts[meta.Category]++
		}

		// Count by question type
		typeCounts[q.QuestionType]++
	}

	stats["category_counts"] = categoryCounts
	stats["type_counts"] = typeCounts

	return stats
}

// Helper function - checks if slice contains string
func containsString(slice []string, item string) bool {
	for _, s := range slice {
		if s == item {
			return true
		}
	}
	return false
}

package handlers

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
)

// WriteJSON writes a JSON response
func WriteJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(payload); err != nil {
		// If encoding fails, we can't write a proper error response
		// since we've already set the status code
		return
	}
}

// WriteJSONFile writes a JSON payload to a file
func WriteJSONFile(path string, payload any) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("create manifest dir: %w", err)
	}
	data, err := json.MarshalIndent(payload, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal json: %w", err)
	}
	return os.WriteFile(path, data, 0o644)
}

// DeriveOCRCommand determines the OCR command to use
func DeriveOCRCommand() []string {
	scriptPath := os.Getenv("DEEPSEEK_OCR_SCRIPT")
	if scriptPath == "" {
		scriptPath = "./scripts/deepseek_ocr_cli.py"
	}

	python := os.Getenv("OCR_PYTHON")
	if python == "" {
		python = "python3"
	}

	scriptAbs, err := filepath.Abs(scriptPath)
	if err != nil {
		return nil
	}

	if _, err := os.Stat(scriptAbs); err != nil {
		return nil
	}

	return []string{python, scriptAbs}
}


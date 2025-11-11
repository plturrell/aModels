package vision

import (
    "context"
    "fmt"
    "time"
)

type DeepSeekConfig struct {
    Domain        string
    Endpoint      string
    APIKey        string
    PythonExec    string
    ScriptPath    string
    ModelVariant  string
    DefaultPrompt string
    Timeout       time.Duration
}

type DeepSeekOCRService struct{
    cfg DeepSeekConfig
}

func NewDeepSeekOCRService(cfg DeepSeekConfig) (*DeepSeekOCRService, error) {
    return &DeepSeekOCRService{cfg: cfg}, nil
}

func (s *DeepSeekOCRService) ExtractText(ctx context.Context, image []byte, prompt string) (string, error) {
    _ = ctx
    _ = image
    if prompt == "" {
        prompt = s.cfg.DefaultPrompt
    }
    return "", fmt.Errorf("DeepSeek OCR not enabled in standalone build")
}



package main

import (
	"context"
	"fmt"
	"net/http"
)

type extractError struct {
	status int
	err    error
}

func (e *extractError) Error() string {
	return e.err.Error()
}

func (e *extractError) Unwrap() error {
	return e.err
}

func (s *extractServer) runExtract(ctx context.Context, req extractRequest) (extractResponse, error) {
	payload, err := s.buildLangextractPayload(req)
	if err != nil {
		return extractResponse{}, &extractError{
			status: http.StatusBadRequest,
			err:    fmt.Errorf("invalid request: %w", err),
		}
	}

	resp, err := s.invokeLangextract(ctx, payload)
	if err != nil {
		return extractResponse{}, &extractError{
			status: http.StatusBadGateway,
			err:    fmt.Errorf("langextract call failed: %w", err),
		}
	}

	if resp.Error != "" {
		return extractResponse{}, &extractError{
			status: http.StatusBadGateway,
			err:    fmt.Errorf("langextract error: %s", resp.Error),
		}
	}

	return extractResponse{
		Entities:    groupExtractions(resp.Extractions),
		Extractions: resp.Extractions,
	}, nil
}

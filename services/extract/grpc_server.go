package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"

	extractpb "github.com/plturrell/aModels/services/extract/gen/extractpb"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/types/known/structpb"
	"google.golang.org/protobuf/types/known/wrapperspb"
)

func (s *extractServer) startGRPCServer(addr string) error {
	listener, err := net.Listen("tcp", addr)
	if err != nil {
		return fmt.Errorf("listen %s: %w", addr, err)
	}

	grpcServer := grpc.NewServer()
	extractpb.RegisterExtractServiceServer(grpcServer, s)

	s.logger.Printf("extract gRPC service listening on %s", addr)

	if err := grpcServer.Serve(listener); err != nil && !errors.Is(err, grpc.ErrServerStopped) {
		return fmt.Errorf("grpc serve: %w", err)
	}
	return nil
}

// Extract implements extractpb.ExtractServiceServer.
func (s *extractServer) Extract(ctx context.Context, req *extractpb.ExtractRequest) (*extractpb.ExtractResponse, error) {
	if req == nil {
		return nil, status.Error(codes.InvalidArgument, "request is required")
	}

	internalReq, err := grpcRequestToInternal(req)
	if err != nil {
		return nil, status.Errorf(codes.InvalidArgument, "invalid request: %v", err)
	}

	internalResp, err := s.runExtract(ctx, internalReq)
	if err != nil {
		return nil, grpcErrorFrom(err)
	}

	grpcResp, err := internalResponseToGRPC(internalResp)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to marshal response: %v", err)
	}
	return grpcResp, nil
}

func grpcRequestToInternal(req *extractpb.ExtractRequest) (extractRequest, error) {
	internal := extractRequest{
		Document:          req.GetDocument(),
		Documents:         append([]string(nil), req.GetDocuments()...),
		PromptDescription: req.GetPromptDescription(),
		ModelID:           req.GetModelId(),
		APIKey:            req.GetApiKey(),
	}

	if val := req.GetTextOrDocuments(); val != nil {
		data, err := protojson.Marshal(val)
		if err != nil {
			return extractRequest{}, fmt.Errorf("marshal text_or_documents: %w", err)
		}
		internal.TextOrDocumentsRaw = json.RawMessage(data)
	}

	if examples := req.GetExamples(); len(examples) > 0 {
		internal.Examples = make([]exampleData, 0, len(examples))
		for _, ex := range examples {
			item := exampleData{
				Text: ex.GetText(),
			}
			if extracted := ex.GetExtractions(); len(extracted) > 0 {
				item.Extractions = make([]exampleExtraction, 0, len(extracted))
				for _, ext := range extracted {
					var attrs map[string]any
					if structAttrs := ext.GetAttributes(); structAttrs != nil {
						attrs = structAttrs.AsMap()
					}
					item.Extractions = append(item.Extractions, exampleExtraction{
						ExtractionClass: ext.GetExtractionClass(),
						ExtractionText:  ext.GetExtractionText(),
						Attributes:      attrs,
					})
				}
			}
			internal.Examples = append(internal.Examples, item)
		}
	}

	return internal, nil
}

func internalResponseToGRPC(resp extractResponse) (*extractpb.ExtractResponse, error) {
	out := &extractpb.ExtractResponse{
		Entities:    make(map[string]*extractpb.EntityList, len(resp.Entities)),
		Extractions: make([]*extractpb.ExtractionResult, 0, len(resp.Extractions)),
	}

	for key, values := range resp.Entities {
		out.Entities[key] = &extractpb.EntityList{Values: append([]string(nil), values...)}
	}

	for _, item := range resp.Extractions {
		var attrs *structpb.Struct
		if len(item.Attributes) > 0 {
			var err error
			attrs, err = structpb.NewStruct(item.Attributes)
			if err != nil {
				return nil, fmt.Errorf("convert attributes: %w", err)
			}
		}

		var start, end *wrapperspb.Int32Value
		if item.StartIndex != nil {
			start = wrapperspb.Int32(int32(*item.StartIndex))
		}
		if item.EndIndex != nil {
			end = wrapperspb.Int32(int32(*item.EndIndex))
		}

		out.Extractions = append(out.Extractions, &extractpb.ExtractionResult{
			ExtractionClass: item.ExtractionClass,
			ExtractionText:  item.ExtractionText,
			Attributes:      attrs,
			StartIndex:      start,
			EndIndex:        end,
		})
	}

	return out, nil
}

func grpcErrorFrom(err error) error {
	var extractErr *extractError
	if errors.As(err, &extractErr) {
		switch extractErr.status {
		case http.StatusBadRequest:
			return status.Error(codes.InvalidArgument, extractErr.Error())
		case http.StatusBadGateway:
			return status.Error(codes.Unavailable, extractErr.Error())
		default:
			return status.Error(codes.Unknown, extractErr.Error())
		}
	}
	return status.Errorf(codes.Internal, "extract failed: %v", err)
}

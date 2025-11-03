package graph

import (
	"bytes"
	"context"
	"encoding/gob"
	"errors"
	"fmt"
	"reflect"

	"github.com/langchain-ai/langgraph-go/pkg/checkpoint"
)

// StateManager handles serialization for checkpoint persistence.
type StateManager struct {
	checkpointer Checkpointer
}

type checkpointEnvelope struct {
	Value any
}

// NewStateManager constructs a manager using the provided checkpointer.
func NewStateManager(cp Checkpointer) *StateManager {
	if cp == nil {
		cp = NoopCheckpointer{}
	}
	return &StateManager{checkpointer: cp}
}

// Save persists state for a node.
func (s *StateManager) Save(ctx context.Context, nodeID NodeID, payload any) error {
	var buf bytes.Buffer
	enc := gob.NewEncoder(&buf)
	if payload != nil {
		gob.Register(payload)
	}
	if err := enc.Encode(checkpointEnvelope{Value: payload}); err != nil {
		return fmt.Errorf("state manager: encode failed: %w", err)
	}
	return s.checkpointer.Save(ctx, string(nodeID), buf.Bytes())
}

// Load restores state for a node into the provided destination. The bool return
// value indicates whether a persisted payload was found.
func (s *StateManager) Load(ctx context.Context, nodeID NodeID, dest any) (bool, error) {
	data, err := s.checkpointer.Load(ctx, string(nodeID))
	if err != nil {
		if errors.Is(err, checkpoint.ErrNotFound) {
			return false, nil
		}
		return false, err
	}
	dec := gob.NewDecoder(bytes.NewReader(data))
	var env checkpointEnvelope
	if err := dec.Decode(&env); err != nil {
		return false, fmt.Errorf("state manager: decode failed: %w", err)
	}
	if env.Value != nil {
		gob.Register(env.Value)
	}
	if err := assignDecoded(dest, env.Value); err != nil {
		return false, err
	}
	return true, nil
}

func assignDecoded(dest any, value any) error {
	if dest == nil {
		return errors.New("state manager: destination must be non-nil")
	}
	dv := reflect.ValueOf(dest)
	if dv.Kind() != reflect.Ptr || dv.IsNil() {
		return errors.New("state manager: destination must be a non-nil pointer")
	}
	target := dv.Elem()
	if value == nil {
		target.Set(reflect.Zero(target.Type()))
		return nil
	}
	v := reflect.ValueOf(value)
	if !v.Type().AssignableTo(target.Type()) {
		if v.Type().ConvertibleTo(target.Type()) {
			v = v.Convert(target.Type())
		} else {
			return fmt.Errorf("state manager: decoded value of type %T cannot be assigned to %s", value, target.Type())
		}
	}
	target.Set(v)
	return nil
}

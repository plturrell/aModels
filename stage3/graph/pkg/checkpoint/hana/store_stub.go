//go:build !hana

package hana

import (
	"context"
	"errors"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_HANA/pkg/hanapool"
)

var errHANANotEnabled = errors.New("hana checkpoint: build without 'hana' tag")

// Option exists for API compatibility when hana support is disabled.
type Option func(*Store)

// Store is a stub used when the hana build tag is not enabled.
type Store struct{}

// NewStore returns an error when hana support is not available.
func NewStore(*hanapool.Pool, ...Option) (*Store, error) { return nil, errHANANotEnabled }

// Save always returns an error when hana support is disabled.
func (*Store) Save(context.Context, string, []byte) error { return errHANANotEnabled }

// Load always returns an error when hana support is disabled.
func (*Store) Load(context.Context, string) ([]byte, error) { return nil, errHANANotEnabled }

// Delete always returns an error when hana support is disabled.
func (*Store) Delete(context.Context, string) error { return errHANANotEnabled }

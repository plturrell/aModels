//go:build !hana

package hana

import (
	"errors"

	hanapool "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_HANA/pkg/hanapool"
)

var errHANABuildTagRequired = errors.New("hana integration: build with -tags hana to enable HANA support")

// NewPoolFromEnv returns an error when the binary is built without the hana tag.
func NewPoolFromEnv() (*hanapool.Pool, error) {
	return nil, errHANABuildTagRequired
}

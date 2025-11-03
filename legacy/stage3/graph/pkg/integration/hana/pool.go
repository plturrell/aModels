//go:build hana

package hana

import (
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_HANA/pkg/hanapool"
)

// NewPoolFromEnv delegates to the shared hanapool implementation to construct a
// connection pool using environment configuration.
func NewPoolFromEnv() (*hanapool.Pool, error) {
	return hanapool.NewPoolFromEnv()
}

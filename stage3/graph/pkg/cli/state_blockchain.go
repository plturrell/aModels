//go:build hana && blockchain

package cli

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/ethereum/go-ethereum/accounts/abi/bind"
	"github.com/ethereum/go-ethereum/common"

	hanastore "github.com/langchain-ai/langgraph-go/pkg/checkpoint/hana"
	"github.com/langchain-ai/langgraph-go/pkg/graph"
	integrationhana "github.com/langchain-ai/langgraph-go/pkg/integration/hana"
	blockchain "github.com/plturrell/agenticAiETH/agenticAiETH_layer1_Blockchain/infrastructure/blockchain"
)

func buildBlockchainCheckpointManager(mode string) (*graph.StateManager, func() error, error) {
	switch mode {
	case "blockchain", "chain":
		return buildBlockchainEnabledHANAStateManager()
	default:
		return nil, nil, fmt.Errorf("unknown hana checkpoint variant %q", mode)
	}
}

func buildBlockchainEnabledHANAStateManager() (*graph.StateManager, func() error, error) {
	ctx := context.Background()

	pool, err := integrationhana.NewPoolFromEnv()
	if err != nil {
		return nil, nil, fmt.Errorf("create hana pool: %w", err)
	}

	resources, err := blockchain.InitBlockchainResources(ctx, blockchain.BlockchainOptions{})
	if err != nil {
		_ = pool.Close()
		return nil, nil, fmt.Errorf("initialise blockchain resources: %w", err)
	}

	registryAddress := strings.TrimSpace(os.Getenv("CHECKPOINT_REGISTRY_ADDRESS"))
	if registryAddress == "" {
		resources.Close()
		_ = pool.Close()
		return nil, nil, fmt.Errorf("CHECKPOINT_REGISTRY_ADDRESS must be configured for blockchain checkpoints")
	}

	registryABI, err := loadCheckpointRegistryABI()
	if err != nil {
		resources.Close()
		_ = pool.Close()
		return nil, nil, err
	}

	registerMethod := strings.TrimSpace(os.Getenv("CHECKPOINT_REGISTRY_REGISTER_METHOD"))
	verifyMethod := strings.TrimSpace(os.Getenv("CHECKPOINT_REGISTRY_VERIFY_METHOD"))

	opts := []hanastore.BlockchainStoreOption{
		hanastore.WithBlockchainEnabled(true),
		hanastore.WithBlockchainClient(resources.EthClient()),
		hanastore.WithRegistryAddress(common.HexToAddress(registryAddress)),
		hanastore.WithRegistryABI(registryABI),
		hanastore.WithTransactOptsProvider(func(callCtx context.Context) (*bind.TransactOpts, error) {
			if err := resources.UpdateGasPrice(callCtx); err != nil {
				log.Printf("warn: failed to refresh gas price for checkpoint registry transaction: %v", err)
			}
			return resources.CloneTransactOpts(callCtx)
		}),
		hanastore.WithCallOptsProvider(func(callCtx context.Context) (*bind.CallOpts, error) {
			return &bind.CallOpts{Context: callCtx, From: resources.AgentAddress()}, nil
		}),
	}

	if registerMethod != "" || verifyMethod != "" {
		if registerMethod == "" {
			registerMethod = "registerCheckpoint"
		}
		if verifyMethod == "" {
			verifyMethod = "verifyCheckpoint"
		}
		opts = append(opts, hanastore.WithRegistryMethods(registerMethod, verifyMethod))
	}

	store, err := hanastore.NewBlockchainVerifiedStore(pool, opts...)
	if err != nil {
		resources.Close()
		_ = pool.Close()
		return nil, nil, err
	}

	cleanup := func() error {
		resources.Close()
		return pool.Close()
	}

	return graph.NewStateManager(store), cleanup, nil
}

func loadCheckpointRegistryABI() (string, error) {
	if path := strings.TrimSpace(os.Getenv("CHECKPOINT_REGISTRY_ABI_FILE")); path != "" {
		expanded := os.ExpandEnv(path)
		if !filepath.IsAbs(expanded) {
			if wd, err := os.Getwd(); err == nil {
				expanded = filepath.Join(wd, expanded)
			}
		}
		data, err := os.ReadFile(expanded) // #nosec G304 - operator supplied path
		if err != nil {
			return "", fmt.Errorf("read CHECKPOINT_REGISTRY_ABI_FILE: %w", err)
		}
		if strings.TrimSpace(string(data)) == "" {
			return "", fmt.Errorf("checkpoint registry ABI file %s is empty", expanded)
		}
		return string(data), nil
	}
	if raw := strings.TrimSpace(os.Getenv("CHECKPOINT_REGISTRY_ABI")); raw != "" {
		return raw, nil
	}
	return "", fmt.Errorf("checkpoint registry ABI not configured; set CHECKPOINT_REGISTRY_ABI or CHECKPOINT_REGISTRY_ABI_FILE")
}

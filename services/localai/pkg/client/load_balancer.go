// Copyright 2025 AgenticAI ETH Contributors
// SPDX-License-Identifier: MIT

package client

import (
	"math/rand"
	"sync"
	"sync/atomic"
	"time"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/types"
)

// LoadBalancer manages load distribution across multiple models
type LoadBalancer struct {
	strategy      types.LoadBalancingStrategy
	roundRobinIdx atomic.Uint64
	modelLoads    map[string]*ModelLoad
	mu            sync.RWMutex
}

// ModelLoad tracks the current load on a model
type ModelLoad struct {
	ActiveRequests atomic.Int64
	TotalRequests  atomic.Int64
	LastLatency    atomic.Int64 // in nanoseconds
}

// NewLoadBalancer creates a new load balancer with the specified strategy
func NewLoadBalancer(strategy types.LoadBalancingStrategy) *LoadBalancer {
	return &LoadBalancer{
		strategy:   strategy,
		modelLoads: make(map[string]*ModelLoad),
	}
}

// SelectModel selects a model from the available options based on the load balancing strategy
func (lb *LoadBalancer) SelectModel(
	candidates []string,
	specs map[string]*types.ModelSpecification,
) string {
	if len(candidates) == 0 {
		return ""
	}

	if len(candidates) == 1 {
		return candidates[0]
	}

	switch lb.strategy {
	case types.StrategyRoundRobin:
		return lb.selectRoundRobin(candidates)
	case types.StrategyLeastLoaded:
		return lb.selectLeastLoaded(candidates)
	case types.StrategyLowestLatency:
		return lb.selectLowestLatency(candidates, specs)
	case types.StrategyRandom:
		return lb.selectRandom(candidates)
	default:
		return lb.selectRoundRobin(candidates)
	}
}

// RecordLatency records the latency for a model
func (lb *LoadBalancer) RecordLatency(model string, latency time.Duration) {
	lb.mu.Lock()
	load, exists := lb.modelLoads[model]
	if !exists {
		load = &ModelLoad{}
		lb.modelLoads[model] = load
	}
	lb.mu.Unlock()

	load.LastLatency.Store(latency.Nanoseconds())
}

// RecordLoad records the current load for a model
func (lb *LoadBalancer) RecordLoad(model string, load int) {
	lb.mu.Lock()
	modelLoad, exists := lb.modelLoads[model]
	if !exists {
		modelLoad = &ModelLoad{}
		lb.modelLoads[model] = modelLoad
	}
	lb.mu.Unlock()

	modelLoad.ActiveRequests.Store(int64(load))
}

// GetLoads returns current load information
func (lb *LoadBalancer) GetLoads() map[string]int {
	lb.mu.RLock()
	defer lb.mu.RUnlock()

	loads := make(map[string]int, len(lb.modelLoads))
	for model, load := range lb.modelLoads {
		loads[model] = int(load.ActiveRequests.Load())
	}
	return loads
}

// RecordRequestStart records the start of a request to a model
func (lb *LoadBalancer) RecordRequestStart(modelName string) {
	lb.mu.Lock()
	load, exists := lb.modelLoads[modelName]
	if !exists {
		load = &ModelLoad{}
		lb.modelLoads[modelName] = load
	}
	lb.mu.Unlock()

	load.ActiveRequests.Add(1)
	load.TotalRequests.Add(1)
}

// RecordRequestEnd records the completion of a request to a model
func (lb *LoadBalancer) RecordRequestEnd(modelName string, latency time.Duration) {
	lb.mu.RLock()
	load, exists := lb.modelLoads[modelName]
	lb.mu.RUnlock()

	if exists {
		load.ActiveRequests.Add(-1)
		load.LastLatency.Store(latency.Nanoseconds())
	}
}

// selectRoundRobin selects models in round-robin fashion
func (lb *LoadBalancer) selectRoundRobin(candidates []string) string {
	idx := lb.roundRobinIdx.Add(1) - 1
	return candidates[idx%uint64(len(candidates))]
}

// selectLeastLoaded selects the model with the fewest active requests
func (lb *LoadBalancer) selectLeastLoaded(candidates []string) string {
	lb.mu.RLock()
	defer lb.mu.RUnlock()

	var selectedModel string
	minLoad := int64(^uint64(0) >> 1) // max int64

	for _, candidate := range candidates {
		load, exists := lb.modelLoads[candidate]
		if !exists {
			// New model with no load - select it
			return candidate
		}

		activeRequests := load.ActiveRequests.Load()
		if activeRequests < minLoad {
			minLoad = activeRequests
			selectedModel = candidate
		}
	}

	if selectedModel == "" {
		return candidates[0]
	}

	return selectedModel
}

// selectLowestLatency selects the model with the lowest average latency
func (lb *LoadBalancer) selectLowestLatency(
	candidates []string,
	specs map[string]*types.ModelSpecification,
) string {
	lb.mu.RLock()
	defer lb.mu.RUnlock()

	var selectedModel string
	minLatency := int64(^uint64(0) >> 1) // max int64

	for _, candidate := range candidates {
		load, exists := lb.modelLoads[candidate]
		if !exists {
			// Use spec's average latency if available
			if spec, hasSpec := specs[candidate]; hasSpec {
				if spec.AverageLatency.Nanoseconds() < minLatency {
					minLatency = spec.AverageLatency.Nanoseconds()
					selectedModel = candidate
				}
			}
			continue
		}

		lastLatency := load.LastLatency.Load()
		if lastLatency < minLatency {
			minLatency = lastLatency
			selectedModel = candidate
		}
	}

	if selectedModel == "" {
		return candidates[0]
	}

	return selectedModel
}

// selectRandom selects a random model from the candidates
func (lb *LoadBalancer) selectRandom(candidates []string) string {
	return candidates[rand.Intn(len(candidates))]
}

// GetModelLoad returns the current load for a model
func (lb *LoadBalancer) GetModelLoad(modelName string) *ModelLoad {
	lb.mu.RLock()
	defer lb.mu.RUnlock()

	load, exists := lb.modelLoads[modelName]
	if !exists {
		return nil
	}

	return load
}

// GetAllLoads returns a snapshot of all model loads
func (lb *LoadBalancer) GetAllLoads() map[string]ModelLoadSnapshot {
	lb.mu.RLock()
	defer lb.mu.RUnlock()

	snapshot := make(map[string]ModelLoadSnapshot, len(lb.modelLoads))
	for modelName, load := range lb.modelLoads {
		snapshot[modelName] = ModelLoadSnapshot{
			ActiveRequests: load.ActiveRequests.Load(),
			TotalRequests:  load.TotalRequests.Load(),
			LastLatency:    time.Duration(load.LastLatency.Load()),
		}
	}

	return snapshot
}

// ModelLoadSnapshot is a point-in-time snapshot of model load
type ModelLoadSnapshot struct {
	ActiveRequests int64
	TotalRequests  int64
	LastLatency    time.Duration
}

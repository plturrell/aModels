package rng

import "math/rand"

// RandFacade provides a narrow surface for random numbers.
type RandFacade struct{ r *rand.Rand }

// NewFacade returns a deterministic RNG with the given seed.
func NewFacade(seed int64) *RandFacade { return &RandFacade{r: rand.New(rand.NewSource(seed))} }

func (rf *RandFacade) Intn(n int) int { return rf.r.Intn(n) }

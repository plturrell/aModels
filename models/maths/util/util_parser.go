package util

import "strconv"

// ParseInt parses a base-10 integer into int.
func ParseInt(s string) (int, error) { return strconv.Atoi(s) }

// ParseFloat parses a float64 value.
func ParseFloat(s string) (float64, error) { return strconv.ParseFloat(s, 64) }

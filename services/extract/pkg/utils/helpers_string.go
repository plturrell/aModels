package utils

// ContainsString checks if a string slice contains a candidate string.
func ContainsString(values []string, candidate string) bool {
	for _, v := range values {
		if v == candidate {
			return true
		}
	}
	return false
}

//go:build hana

package storage

import "strings"

func isSyntaxError(err error) bool {
	if err == nil {
		return false
	}
	msg := strings.ToLower(err.Error())
	return strings.Contains(msg, "syntax error") || strings.Contains(msg, "near \"if\"")
}

func isAlreadyExistsError(err error) bool {
	if err == nil {
		return false
	}
	msg := strings.ToLower(err.Error())
	return strings.Contains(msg, "already exists") ||
		strings.Contains(msg, "duplicate column") ||
		strings.Contains(msg, "duplicate table") ||
		strings.Contains(msg, "duplicate index")
}

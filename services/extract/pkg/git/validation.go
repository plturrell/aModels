package git

import (
	"fmt"
	"path/filepath"
	"regexp"
	"strings"
)

// Validation errors
var (
	ErrInvalidRepoName   = fmt.Errorf("repository name must contain only alphanumeric characters, dots, hyphens, and underscores")
	ErrInvalidOwnerName  = fmt.Errorf("owner name must contain only alphanumeric characters, hyphens, and underscores")
	ErrInvalidBranchName = fmt.Errorf("branch name must contain only alphanumeric characters, forward slashes, dots, hyphens, and underscores")
	ErrInvalidFilePath   = fmt.Errorf("file path contains invalid characters or path traversal attempts")
	ErrDescriptionTooLong = fmt.Errorf("description exceeds maximum length of 2000 characters")
)

var (
	repoNameRegex   = regexp.MustCompile(`^[a-zA-Z0-9._-]+$`)
	ownerNameRegex  = regexp.MustCompile(`^[a-zA-Z0-9._-]+$`)
	branchNameRegex = regexp.MustCompile(`^[a-zA-Z0-9._/-]+$`)
)

// ValidateRepositoryName validates a repository name.
// Repository names must contain only alphanumeric characters, dots, hyphens, and underscores.
// Maximum length is 100 characters. Empty names are not allowed.
func ValidateRepositoryName(name string) error {
	if name == "" {
		return fmt.Errorf("repository name cannot be empty")
	}
	if len(name) > 100 {
		return fmt.Errorf("repository name exceeds maximum length of 100 characters")
	}
	if !repoNameRegex.MatchString(name) {
		return ErrInvalidRepoName
	}
	return nil
}

// ValidateOwnerName validates an owner name (user or organization).
// Owner names must contain only alphanumeric characters, dots, hyphens, and underscores.
// Maximum length is 100 characters. Empty owner is allowed (for user repositories).
func ValidateOwnerName(owner string) error {
	if owner == "" {
		return nil // Owner can be empty (user repos)
	}
	if len(owner) > 100 {
		return fmt.Errorf("owner name exceeds maximum length of 100 characters")
	}
	if !ownerNameRegex.MatchString(owner) {
		return ErrInvalidOwnerName
	}
	return nil
}

// ValidateBranchName validates a Git branch name.
// Branch names must contain only alphanumeric characters, forward slashes, dots, hyphens, and underscores.
// Maximum length is 255 characters. Cannot start or end with dots, and cannot contain consecutive dots.
func ValidateBranchName(branch string) error {
	if branch == "" {
		return fmt.Errorf("branch name cannot be empty")
	}
	if len(branch) > 255 {
		return fmt.Errorf("branch name exceeds maximum length of 255 characters")
	}
	if !branchNameRegex.MatchString(branch) {
		return ErrInvalidBranchName
	}
	// Git branch name restrictions
	if strings.HasPrefix(branch, ".") || strings.HasSuffix(branch, ".") {
		return fmt.Errorf("branch name cannot start or end with a dot")
	}
	if strings.Contains(branch, "..") {
		return fmt.Errorf("branch name cannot contain consecutive dots")
	}
	return nil
}

// ValidateFilePath validates and sanitizes a file path.
// Prevents path traversal attacks by rejecting paths containing ".." or starting with "../".
// Rejects absolute paths and paths containing null characters.
// Returns the normalized path with forward slashes, or an error if validation fails.
func ValidateFilePath(filePath string) (string, error) {
	if filePath == "" {
		return "", fmt.Errorf("file path cannot be empty")
	}
	
	// Normalize path
	normalized := filepath.Clean(filePath)
	
	// Prevent path traversal
	if strings.HasPrefix(normalized, "..") || strings.Contains(normalized, "../") {
		return "", ErrInvalidFilePath
	}
	
	// Prevent absolute paths
	if filepath.IsAbs(normalized) {
		return "", fmt.Errorf("file path cannot be absolute")
	}
	
	// Check for invalid characters (basic check)
	if strings.Contains(normalized, "\x00") {
		return "", fmt.Errorf("file path contains null character")
	}
	
	// Convert to forward slashes for Gitea API
	normalized = filepath.ToSlash(normalized)
	
	return normalized, nil
}

// ValidateDescription validates a repository description.
// Maximum length is 2000 characters. Empty descriptions are allowed.
func ValidateDescription(description string) error {
	if len(description) > 2000 {
		return ErrDescriptionTooLong
	}
	return nil
}

// SanitizeOwnerRepo validates and sanitizes both owner and repository names.
// Returns the validated owner and repo names, or an error if validation fails.
func SanitizeOwnerRepo(owner, repo string) (string, string, error) {
	if err := ValidateOwnerName(owner); err != nil {
		return "", "", err
	}
	if err := ValidateRepositoryName(repo); err != nil {
		return "", "", err
	}
	return owner, repo, nil
}


package git

import "testing"

func TestValidateRepositoryName(t *testing.T) {
	tests := []struct {
		name    string
		wantErr bool
	}{
		{"valid-repo", false},
		{"valid_repo", false},
		{"valid.repo", false},
		{"ValidRepo123", false},
		{"", true}, // Empty name
		{"repo with spaces", true},
		{"repo@invalid", true},
		{"repo#invalid", true},
		{string(make([]byte, 101)), true}, // Too long
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateRepositoryName(tt.name)
			if (err != nil) != tt.wantErr {
				t.Errorf("ValidateRepositoryName(%q) error = %v, wantErr %v", tt.name, err, tt.wantErr)
			}
		})
	}
}

func TestValidateOwnerName(t *testing.T) {
	tests := []struct {
		owner   string
		wantErr bool
	}{
		{"valid-owner", false},
		{"valid_owner", false},
		{"valid.owner", false},
		{"", false}, // Empty is allowed (user repos)
		{"owner with spaces", true},
		{"owner@invalid", true},
		{string(make([]byte, 101)), true}, // Too long
	}
	
	for _, tt := range tests {
		t.Run(tt.owner, func(t *testing.T) {
			err := ValidateOwnerName(tt.owner)
			if (err != nil) != tt.wantErr {
				t.Errorf("ValidateOwnerName(%q) error = %v, wantErr %v", tt.owner, err, tt.wantErr)
			}
		})
	}
}

func TestValidateBranchName(t *testing.T) {
	tests := []struct {
		branch  string
		wantErr bool
	}{
		{"main", false},
		{"develop", false},
		{"feature/new-feature", false},
		{"release/v1.0", false},
		{"", true}, // Empty branch
		{".invalid", true}, // Starts with dot
		{"invalid.", true}, // Ends with dot
		{"invalid..name", true}, // Consecutive dots
		{string(make([]byte, 256)), true}, // Too long
	}
	
	for _, tt := range tests {
		t.Run(tt.branch, func(t *testing.T) {
			err := ValidateBranchName(tt.branch)
			if (err != nil) != tt.wantErr {
				t.Errorf("ValidateBranchName(%q) error = %v, wantErr %v", tt.branch, err, tt.wantErr)
			}
		})
	}
}

func TestValidateFilePath(t *testing.T) {
	tests := []struct {
		path    string
		wantErr bool
	}{
		{"file.txt", false},
		{"path/to/file.txt", false},
		{"path/to/file", false},
		{"", true}, // Empty path
		{"../invalid", true}, // Path traversal
		{"../../etc/passwd", true}, // Path traversal
		{"/absolute/path", true}, // Absolute path
		{"path\x00withnull", true}, // Null character
	}
	
	for _, tt := range tests {
		t.Run(tt.path, func(t *testing.T) {
			_, err := ValidateFilePath(tt.path)
			if (err != nil) != tt.wantErr {
				t.Errorf("ValidateFilePath(%q) error = %v, wantErr %v", tt.path, err, tt.wantErr)
			}
		})
	}
}

func TestValidateDescription(t *testing.T) {
	tests := []struct {
		desc    string
		wantErr bool
	}{
		{"Valid description", false},
		{"", false}, // Empty is allowed
		{string(make([]byte, 2001)), true}, // Too long
	}
	
	for _, tt := range tests {
		t.Run("description", func(t *testing.T) {
			err := ValidateDescription(tt.desc)
			if (err != nil) != tt.wantErr {
				t.Errorf("ValidateDescription() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}


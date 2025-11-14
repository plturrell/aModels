package git

import (
	"regexp"
	"strings"
)

// ContentScanner scans code content for secrets and sensitive information
type ContentScanner struct {
	patterns []SecretPattern
}

// SecretPattern represents a pattern for detecting secrets
type SecretPattern struct {
	Name        string
	Pattern     *regexp.Regexp
	Description string
	Severity    string // "high", "medium", "low"
}

// NewContentScanner creates a new content scanner with default patterns
func NewContentScanner() *ContentScanner {
	patterns := []SecretPattern{
		{
			Name:        "API Key",
			Pattern:     regexp.MustCompile(`(?i)(api[_-]?key|apikey)\s*[:=]\s*["']?([a-zA-Z0-9_\-]{20,})["']?`),
			Description: "Potential API key detected",
			Severity:    "high",
		},
		{
			Name:        "AWS Access Key",
			Pattern:     regexp.MustCompile(`AKIA[0-9A-Z]{16}`),
			Description: "AWS access key ID detected",
			Severity:    "high",
		},
		{
			Name:        "Private Key",
			Pattern:     regexp.MustCompile(`-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----`),
			Description: "Private key detected",
			Severity:    "high",
		},
		{
			Name:        "Password",
			Pattern:     regexp.MustCompile(`(?i)(password|passwd|pwd)\s*[:=]\s*["']?([^\s"']{8,})["']?`),
			Description: "Potential password detected",
			Severity:    "medium",
		},
		{
			Name:        "Token",
			Pattern:     regexp.MustCompile(`(?i)(token|bearer)\s*[:=]\s*["']?([a-zA-Z0-9_\-]{20,})["']?`),
			Description: "Potential token detected",
			Severity:    "high",
		},
		{
			Name:        "Database Connection String",
			Pattern:     regexp.MustCompile(`(?i)(mongodb|postgres|mysql|redis)://[^\s"']+`),
			Description: "Database connection string detected",
			Severity:    "high",
		},
		{
			Name:        "Email",
			Pattern:     regexp.MustCompile(`[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}`),
			Description: "Email address detected (potential PII)",
			Severity:    "low",
		},
		{
			Name:        "Credit Card",
			Pattern:     regexp.MustCompile(`\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b`),
			Description: "Potential credit card number detected",
			Severity:    "high",
		},
	}

	return &ContentScanner{
		patterns: patterns,
	}
}

// ScanResult represents the result of scanning content
type ScanResult struct {
	HasSecrets bool
	Findings   []Finding
	RiskLevel  string // "high", "medium", "low", "none"
}

// Finding represents a single secret finding
type Finding struct {
	Type        string
	Description string
	Severity    string
	Line        int
	Match       string
}

// Scan scans content for secrets and sensitive information
func (s *ContentScanner) Scan(content string) *ScanResult {
	var findings []Finding
	lines := strings.Split(content, "\n")

	for lineNum, line := range lines {
		for _, pattern := range s.patterns {
			matches := pattern.Pattern.FindAllStringSubmatch(line, -1)
			for _, match := range matches {
				if len(match) > 0 {
					findings = append(findings, Finding{
						Type:        pattern.Name,
						Description: pattern.Description,
						Severity:    pattern.Severity,
						Line:        lineNum + 1,
						Match:       maskSecret(match[0]),
					})
				}
			}
		}
	}

	riskLevel := determineRiskLevel(findings)

	return &ScanResult{
		HasSecrets: len(findings) > 0,
		Findings:   findings,
		RiskLevel:  riskLevel,
	}
}

// maskSecret masks sensitive parts of a secret
func maskSecret(secret string) string {
	if len(secret) <= 8 {
		return "***"
	}
	return secret[:4] + "***" + secret[len(secret)-4:]
}

// determineRiskLevel determines the overall risk level from findings
func determineRiskLevel(findings []Finding) string {
	hasHigh := false
	hasMedium := false

	for _, finding := range findings {
		switch finding.Severity {
		case "high":
			hasHigh = true
		case "medium":
			hasMedium = true
		}
	}

	if hasHigh {
		return "high"
	}
	if hasMedium {
		return "medium"
	}
	return "low"
}


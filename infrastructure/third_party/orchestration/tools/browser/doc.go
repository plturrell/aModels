// Package browser provides browser automation tools for the Orchestration framework.
//
// The browser package implements web automation capabilities including:
// - Web page navigation
// - Data extraction using CSS selectors
// - AI-powered content analysis
// - Form interaction (click, type)
// - Multi-step workflow automation
// - Privacy-preserving data collection
//
// Example usage:
//
//	// Create a browser tool
//	browserTool := browser.NewBrowserTool("http://localhost:9222")
//
//	// Use in an agent
//	agent := agents.NewReActAgent(llm, []tools.Tool{browserTool})
//
//	// Agent can now browse and extract data
//	result, err := agent.Run(ctx, "Navigate to example.com and extract the main heading")
//
// The browser tool integrates with:
// - LocalAI for intelligent page analysis
// - HANA for privacy-preserving data storage
// - Differential privacy for data protection
//
// Privacy features:
// - Automatic PII detection and removal
// - Differential privacy noise addition
// - Privacy budget tracking
// - IP address anonymization
package browser

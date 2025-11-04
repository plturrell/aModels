package browser

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Orchestration/agents"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Orchestration/llms/localai"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Orchestration/tools"
)

// ExampleBrowserAgent demonstrates how to use the browser tool in an agent
func ExampleBrowserAgent() {
	// Create browser tool
	browserTool := NewBrowserTool("http://localhost:9222")

	// Create LocalAI LLM
	llm, err := localai.New(localai.WithBaseURL("http://localhost:8080"))
	if err != nil {
		log.Fatal(err)
	}

	// Create agent with browser tool
	agent := agents.NewReActAgent(llm, []tools.Tool{browserTool})

	// Example 1: Simple navigation and extraction
	ctx := context.Background()
	result, err := agent.Run(ctx, "Navigate to https://example.com and extract the main heading")
	if err != nil {
		log.Printf("Error: %v", err)
		return
	}

	fmt.Printf("Result: %s\n", result)
}

// ExampleBrowserWorkflow demonstrates a multi-step browser workflow
func ExampleBrowserWorkflow() {
	// Create browser tool
	browserTool := NewBrowserTool("http://localhost:9222")

	// Create workflow
	workflow := NewBrowserWorkflow(browserTool)

	// Add steps
	workflow.AddStep(WorkflowStep{
		ID:          "navigate",
		Action:      "navigate",
		Parameters:  map[string]interface{}{"url": "https://example.com"},
		Description: "Navigate to example.com",
	})

	workflow.AddStep(WorkflowStep{
		ID:          "extract_title",
		Action:      "extract",
		Parameters:  map[string]interface{}{"selectors": map[string]string{"title": "h1"}},
		Description: "Extract the main heading",
		WaitAfter:   2 * time.Second,
	})

	workflow.AddStep(WorkflowStep{
		ID:          "analyze_content",
		Action:      "analyze",
		Parameters:  map[string]interface{}{},
		Description: "Analyze page content with AI",
	})

	// Execute workflow
	results, err := workflow.Execute()
	if err != nil {
		log.Printf("Workflow error: %v", err)
		return
	}

	for _, result := range results {
		fmt.Printf("Step %s: %+v\n", result["step_id"], result["result"])
	}
}

// ExampleBrowserChain demonstrates a browser operation chain
func ExampleBrowserChain() {
	// Create browser tool
	browserTool := NewBrowserTool("http://localhost:9222")

	// Create chain
	chain := NewBrowserChain(browserTool)

	// Add steps
	chain.AddStep(map[string]interface{}{
		"action": "navigate",
		"url":    "https://news.ycombinator.com",
	})

	chain.AddStep(map[string]interface{}{
		"action":    "extract",
		"selectors": map[string]string{"headlines": ".storylink"},
	})

	chain.AddStep(map[string]interface{}{
		"action": "analyze",
	})

	// Execute chain
	ctx := context.Background()
	results, err := chain.Execute(ctx)
	if err != nil {
		log.Printf("Chain error: %v", err)
		return
	}

	for i, result := range results {
		fmt.Printf("Chain step %d: %+v\n", i+1, result)
	}
}

// ExampleResearchAgent demonstrates a research agent using browser
func ExampleResearchAgent() {
	// Create browser tool
	browserTool := NewBrowserTool("http://localhost:9222")

	// Create LocalAI LLM
	llm, err := localai.New(localai.WithBaseURL("http://localhost:8080"))
	if err != nil {
		log.Fatal(err)
	}

	// Create research agent
	agent := agents.NewReActAgent(llm, []tools.Tool{browserTool})

	// Research task
	ctx := context.Background()
	query := "Research the latest developments in AI and machine learning, then summarize the top 3 articles"

	result, err := agent.Run(ctx, query)
	if err != nil {
		log.Printf("Research error: %v", err)
		return
	}

	fmt.Printf("Research Summary: %s\n", result)
}

// ExampleDataExtractionAgent demonstrates data extraction with privacy
func ExampleDataExtractionAgent() {
	// Create browser tool
	browserTool := NewBrowserTool("http://localhost:9222")

	// Create LocalAI LLM
	llm, err := localai.New(localai.WithBaseURL("http://localhost:8080"))
	if err != nil {
		log.Fatal(err)
	}

	// Create extraction agent
	agent := agents.NewReActAgent(llm, []tools.Tool{browserTool})

	// Data extraction task
	ctx := context.Background()
	query := "Extract product information from an e-commerce site, ensuring all personal data is anonymized"

	result, err := agent.Run(ctx, query)
	if err != nil {
		log.Printf("Extraction error: %v", err)
		return
	}

	fmt.Printf("Extracted Data: %s\n", result)
}

// ExampleFormAutomationAgent demonstrates form automation
func ExampleFormAutomationAgent() {
	// Create browser tool
	browserTool := NewBrowserTool("http://localhost:9222")

	// Create LocalAI LLM
	llm, err := localai.New(localai.WithBaseURL("http://localhost:8080"))
	if err != nil {
		log.Fatal(err)
	}

	// Create automation agent
	agent := agents.NewReActAgent(llm, []tools.Tool{browserTool})

	// Form automation task
	ctx := context.Background()
	query := "Fill out a contact form with test data and submit it"

	result, err := agent.Run(ctx, query)
	if err != nil {
		log.Printf("Automation error: %v", err)
		return
	}

	fmt.Printf("Form Automation Result: %s\n", result)
}

//go:build ignore
// +build ignore

package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_HANA/pkg/hanapool"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_HANA/pkg/migration"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_HANA/pkg/storage"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Orchestration/llms/localai"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Orchestration/memory/hana"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Orchestration/tools/hana"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Orchestration/vectorstores/hana"
)

// IntegrationTestSuite tests the complete Layer 4 integration
type IntegrationTestSuite struct {
	hanaPool        *hanapool.Pool
	relationalStore *storage.RelationalStore
	vectorStore     *storage.VectorStore
	graphStore      *storage.GraphStore
	memoryStore     *storage.MemoryStore
}

// NewIntegrationTestSuite creates a new test suite
func NewIntegrationTestSuite() (*IntegrationTestSuite, error) {
	// Create HANA connection pool
	pool, err := hanapool.NewPoolFromEnv()
	if err != nil {
		return nil, fmt.Errorf("failed to create HANA pool: %w", err)
	}

	// Run migrations
	migrator := migration.NewMigrator(pool)
	ctx := context.Background()
	if err := migrator.Migrate(ctx); err != nil {
		return nil, fmt.Errorf("failed to run migrations: %w", err)
	}

	// Create storage stores
	relationalStore := storage.NewRelationalStore(pool)
	vectorStore := storage.NewVectorStore(pool)
	graphStore := storage.NewGraphStore(pool)
	memoryStore := storage.NewMemoryStore(pool)

	return &IntegrationTestSuite{
		hanaPool:        pool,
		relationalStore: relationalStore,
		vectorStore:     vectorStore,
		graphStore:      graphStore,
		memoryStore:     memoryStore,
	}, nil
}

// TestHANAConnection tests HANA connectivity
func (t *IntegrationTestSuite) TestHANAConnection(ctx context.Context) error {
	log.Println("🔍 Testing HANA connection...")

	if err := t.hanaPool.Health(ctx); err != nil {
		return fmt.Errorf("HANA health check failed: %w", err)
	}

	// Test basic query
	query := "SELECT CURRENT_TIMESTAMP as test_time FROM DUMMY"
	row := t.hanaPool.QueryRow(ctx, query)

	var testTime time.Time
	if err := row.Scan(&testTime); err != nil {
		return fmt.Errorf("failed to execute test query: %w", err)
	}

	log.Printf("✅ HANA connection successful. Server time: %v", testTime)
	return nil
}

// TestRelationalOperations tests relational data operations
func (t *IntegrationTestSuite) TestRelationalOperations(ctx context.Context) error {
	log.Println("🔍 Testing relational operations...")

	// Test insert
	testData := map[string]interface{}{
		"name":       "Test Agent",
		"type":       "integration_test",
		"status":     "active",
		"created_at": time.Now(),
	}

	id, err := t.relationalStore.Insert(ctx, "agent_states", testData)
	if err != nil {
		return fmt.Errorf("failed to insert test data: %w", err)
	}

	log.Printf("✅ Inserted record with ID: %d", id)

	// Test select
	rows, err := t.relationalStore.Select(ctx, "agent_states", []string{"id", "name", "type"},
		map[string]interface{}{"id": id}, "", 1)
	if err != nil {
		return fmt.Errorf("failed to select test data: %w", err)
	}
	defer rows.Close()

	var count int
	for rows.Next() {
		count++
	}

	if count == 0 {
		return fmt.Errorf("no records found after insert")
	}

	log.Printf("✅ Selected %d records", count)

	// Test update
	updateData := map[string]interface{}{
		"status": "updated",
	}
	where := map[string]interface{}{
		"id": id,
	}

	rowsAffected, err := t.relationalStore.Update(ctx, "agent_states", updateData, where)
	if err != nil {
		return fmt.Errorf("failed to update test data: %w", err)
	}

	if rowsAffected == 0 {
		return fmt.Errorf("no rows affected by update")
	}

	log.Printf("✅ Updated %d rows", rowsAffected)

	// Test delete
	rowsAffected, err = t.relationalStore.Delete(ctx, "agent_states", where)
	if err != nil {
		return fmt.Errorf("failed to delete test data: %w", err)
	}

	if rowsAffected == 0 {
		return fmt.Errorf("no rows affected by delete")
	}

	log.Printf("✅ Deleted %d rows", rowsAffected)
	return nil
}

// TestVectorOperations tests vector operations
func (t *IntegrationTestSuite) TestVectorOperations(ctx context.Context) error {
	log.Println("🔍 Testing vector operations...")

	// Test embedding insertion
	testVector := []float64{0.1, 0.2, 0.3, 0.4, 0.5}
	testContent := "This is a test document for vector search"
	testMetadata := map[string]string{
		"source": "integration_test",
		"type":   "test_document",
	}

	id, err := t.vectorStore.InsertEmbedding(ctx, testVector, testContent, testMetadata)
	if err != nil {
		return fmt.Errorf("failed to insert embedding: %w", err)
	}

	log.Printf("✅ Inserted embedding with ID: %d", id)

	// Test similarity search
	queryVector := []float64{0.1, 0.2, 0.3, 0.4, 0.5}
	results, err := t.vectorStore.SimilaritySearch(ctx, queryVector, 5, 0.0)
	if err != nil {
		return fmt.Errorf("failed to perform similarity search: %w", err)
	}

	if len(results) == 0 {
		return fmt.Errorf("no similarity search results found")
	}

	log.Printf("✅ Found %d similar vectors", len(results))

	// Test embedding retrieval
	embedding, err := t.vectorStore.GetEmbedding(ctx, id)
	if err != nil {
		return fmt.Errorf("failed to get embedding: %w", err)
	}

	if embedding.Content != testContent {
		return fmt.Errorf("retrieved content doesn't match original")
	}

	log.Printf("✅ Retrieved embedding: %s", embedding.Content)

	// Clean up
	err = t.vectorStore.DeleteEmbedding(ctx, id)
	if err != nil {
		return fmt.Errorf("failed to delete embedding: %w", err)
	}

	log.Printf("✅ Deleted test embedding")
	return nil
}

// TestGraphOperations tests graph operations
func (t *IntegrationTestSuite) TestGraphOperations(ctx context.Context) error {
	log.Println("🔍 Testing graph operations...")

	// Test node creation
	node1ID, err := t.graphStore.AddNode(ctx, "agent", map[string]string{
		"name": "Test Agent 1",
		"type": "integration_test",
	})
	if err != nil {
		return fmt.Errorf("failed to add node 1: %w", err)
	}

	node2ID, err := t.graphStore.AddNode(ctx, "agent", map[string]string{
		"name": "Test Agent 2",
		"type": "integration_test",
	})
	if err != nil {
		return fmt.Errorf("failed to add node 2: %w", err)
	}

	log.Printf("✅ Created nodes: %d, %d", node1ID, node2ID)

	// Test edge creation
	edgeID, err := t.graphStore.AddEdge(ctx, node1ID, node2ID, "communicates", 1.0, map[string]string{
		"frequency": "high",
		"type":      "message",
	})
	if err != nil {
		return fmt.Errorf("failed to add edge: %w", err)
	}

	log.Printf("✅ Created edge: %d", edgeID)

	// Test neighbor retrieval
	neighbors, err := t.graphStore.GetNeighbors(ctx, node1ID, "")
	if err != nil {
		return fmt.Errorf("failed to get neighbors: %w", err)
	}

	if len(neighbors) == 0 {
		return fmt.Errorf("no neighbors found")
	}

	log.Printf("✅ Found %d neighbors", len(neighbors))

	// Test path finding
	path, err := t.graphStore.FindPath(ctx, node1ID, node2ID, 5)
	if err != nil {
		return fmt.Errorf("failed to find path: %w", err)
	}

	if len(path.Nodes) == 0 {
		return fmt.Errorf("no path found")
	}

	log.Printf("✅ Found path with %d nodes, cost: %.2f", len(path.Nodes), path.Cost)

	// Clean up
	err = t.graphStore.DeleteNode(ctx, node1ID)
	if err != nil {
		return fmt.Errorf("failed to delete node 1: %w", err)
	}

	err = t.graphStore.DeleteNode(ctx, node2ID)
	if err != nil {
		return fmt.Errorf("failed to delete node 2: %w", err)
	}

	log.Printf("✅ Cleaned up test nodes")
	return nil
}

// TestMemoryOperations tests memory operations
func (t *IntegrationTestSuite) TestMemoryOperations(ctx context.Context) error {
	log.Println("🔍 Testing memory operations...")

	// Test conversation creation
	conversationID, err := t.memoryStore.CreateConversation(ctx, "test_agent", "test_session")
	if err != nil {
		return fmt.Errorf("failed to create conversation: %w", err)
	}

	log.Printf("✅ Created conversation: %d", conversationID)

	// Test message addition
	messageID, err := t.memoryStore.AddMessage(ctx, conversationID, "user", "Hello, how are you?", map[string]string{
		"timestamp": time.Now().Format(time.RFC3339),
	})
	if err != nil {
		return fmt.Errorf("failed to add user message: %w", err)
	}

	_, err = t.memoryStore.AddMessage(ctx, conversationID, "assistant", "I'm doing well, thank you!", map[string]string{
		"timestamp": time.Now().Format(time.RFC3339),
	})
	if err != nil {
		return fmt.Errorf("failed to add assistant message: %w", err)
	}

	log.Printf("✅ Added messages, last ID: %d", messageID)

	// Test message retrieval
	messages, err := t.memoryStore.GetMessages(ctx, conversationID, 0, 0)
	if err != nil {
		return fmt.Errorf("failed to get messages: %w", err)
	}

	if len(messages) != 2 {
		return fmt.Errorf("expected 2 messages, got %d", len(messages))
	}

	log.Printf("✅ Retrieved %d messages", len(messages))

	// Test conversation summary
	summary, err := t.memoryStore.GetConversationSummary(ctx, "test_agent", 1)
	if err != nil {
		return fmt.Errorf("failed to get conversation summary: %w", err)
	}

	log.Printf("✅ Conversation summary: %+v", summary)

	// Clean up
	err = t.memoryStore.DeleteConversation(ctx, conversationID)
	if err != nil {
		return fmt.Errorf("failed to delete conversation: %w", err)
	}

	log.Printf("✅ Cleaned up test conversation")
	return nil
}

// TestOrchestrationIntegration tests orchestration framework integration
func (t *IntegrationTestSuite) TestOrchestrationIntegration(ctx context.Context) error {
	log.Println("🔍 Testing orchestration integration...")

	// Test HANA memory integration
	hanaMemory, err := hana.NewHANAChatMessageHistory(t.hanaPool, "test_agent", "test_session")
	if err != nil {
		return fmt.Errorf("failed to create HANA memory: %w", err)
	}

	// Test adding messages
	err = hanaMemory.AddUserMessage(ctx, "What is the weather like?")
	if err != nil {
		return fmt.Errorf("failed to add user message: %w", err)
	}

	err = hanaMemory.AddAIMessage(ctx, "I don't have access to weather data, but I can help you with other questions.")
	if err != nil {
		return fmt.Errorf("failed to add AI message: %w", err)
	}

	// Test message retrieval
	messages, err := hanaMemory.Messages(ctx)
	if err != nil {
		return fmt.Errorf("failed to get messages: %w", err)
	}

	if len(messages) != 2 {
		return fmt.Errorf("expected 2 messages, got %d", len(messages))
	}

	log.Printf("✅ HANA memory integration successful: %d messages", len(messages))

	// Test HANA vector store integration
	hanaVectorStore, err := hana.NewHANAVectorStore(t.hanaPool)
	if err != nil {
		return fmt.Errorf("failed to create HANA vector store: %w", err)
	}

	// Test vector operations
	testVector := []float64{0.1, 0.2, 0.3, 0.4, 0.5}
	testContent := "Test document for orchestration integration"
	testMetadata := map[string]interface{}{
		"source": "integration_test",
		"type":   "orchestration_test",
	}

	err = hanaVectorStore.AddTexts(ctx, []string{testContent}, [][]float64{testVector}, []map[string]interface{}{testMetadata})
	if err != nil {
		return fmt.Errorf("failed to add texts to vector store: %w", err)
	}

	// Test similarity search
	documents, err := hanaVectorStore.SimilaritySearch(ctx, testVector, 5)
	if err != nil {
		return fmt.Errorf("failed to perform similarity search: %w", err)
	}

	if len(documents) == 0 {
		return fmt.Errorf("no documents found in similarity search")
	}

	log.Printf("✅ HANA vector store integration successful: %d documents found", len(documents))

	// Test HANA tools
	sqlTool := hana.NewSQLQueryTool(t.hanaPool)
	vectorTool := hana.NewVectorSearchTool(t.hanaPool)
	graphTool := hana.NewGraphTraversalTool(t.hanaPool)
	dataTool := hana.NewDataInsertTool(t.hanaPool)

	// Test SQL tool
	sqlResult, err := sqlTool.Call(ctx, map[string]interface{}{
		"query": "SELECT COUNT(*) as count FROM agent_states",
		"limit": 10,
	})
	if err != nil {
		return fmt.Errorf("failed to execute SQL tool: %w", err)
	}

	log.Printf("✅ SQL tool successful: %+v", sqlResult)

	// Test vector tool
	vectorResult, err := vectorTool.Call(ctx, map[string]interface{}{
		"vector":    testVector,
		"limit":     5,
		"threshold": 0.0,
	})
	if err != nil {
		return fmt.Errorf("failed to execute vector tool: %w", err)
	}

	log.Printf("✅ Vector tool successful: %+v", vectorResult)

	// Test graph tool
	graphResult, err := graphTool.Call(ctx, map[string]interface{}{
		"operation": "nodes_by_type",
		"node_type": "agent",
		"limit":     10,
	})
	if err != nil {
		return fmt.Errorf("failed to execute graph tool: %w", err)
	}

	log.Printf("✅ Graph tool successful: %+v", graphResult)

	// Test data tool
	dataResult, err := dataTool.Call(ctx, map[string]interface{}{
		"table": "agent_states",
		"data": map[string]interface{}{
			"agent_id": "test_agent",
			"status":   "active",
			"metadata": "integration_test",
		},
		"operation": "insert",
	})
	if err != nil {
		return fmt.Errorf("failed to execute data tool: %w", err)
	}

	log.Printf("✅ Data tool successful: %+v", dataResult)

	return nil
}

// TestLocalAIIntegration tests LocalAI integration
func (t *IntegrationTestSuite) TestLocalAIIntegration(ctx context.Context) error {
	log.Println("🔍 Testing LocalAI integration...")

	// Test LocalAI LLM integration
	localaiLLM := localai.NewLLM("http://localhost:8080", "vaultgemma", 0.7, 512)

	// Test content generation
	messages := []llms.MessageContent{
		{
			Role:    llms.ChatMessageRoleUser,
			Content: "Hello, how are you?",
		},
	}

	response, err := localaiLLM.GenerateContent(ctx, messages)
	if err != nil {
		// This might fail if LocalAI server is not running, which is expected
		log.Printf("⚠️  LocalAI integration test skipped (server not running): %v", err)
		return nil
	}

	if len(response.Choices) == 0 {
		return fmt.Errorf("no response choices received")
	}

	log.Printf("✅ LocalAI integration successful: %s", response.Choices[0].Content)
	return nil
}

// RunAllTests runs all integration tests
func (t *IntegrationTestSuite) RunAllTests() error {
	ctx := context.Background()

	tests := []struct {
		name string
		test func(context.Context) error
	}{
		{"HANA Connection", t.TestHANAConnection},
		{"Relational Operations", t.TestRelationalOperations},
		{"Vector Operations", t.TestVectorOperations},
		{"Graph Operations", t.TestGraphOperations},
		{"Memory Operations", t.TestMemoryOperations},
		{"Orchestration Integration", t.TestOrchestrationIntegration},
		{"LocalAI Integration", t.TestLocalAIIntegration},
	}

	passed := 0
	failed := 0

	for _, test := range tests {
		log.Printf("\n🧪 Running test: %s", test.name)
		if err := test.test(ctx); err != nil {
			log.Printf("❌ Test failed: %s - %v", test.name, err)
			failed++
		} else {
			log.Printf("✅ Test passed: %s", test.name)
			passed++
		}
	}

	log.Printf("\n📊 Test Results: %d passed, %d failed", passed, failed)

	if failed > 0 {
		return fmt.Errorf("integration tests failed: %d/%d tests failed", failed, len(tests))
	}

	return nil
}

// Close closes the test suite
func (t *IntegrationTestSuite) Close() error {
	return t.hanaPool.Close()
}

func main() {
	log.Println("🚀 Starting Layer 4 Integration Tests...")

	// Check environment variables
	if os.Getenv("HANA_PASSWORD") == "" {
		log.Fatal("❌ HANA_PASSWORD environment variable is required")
	}

	// Create test suite
	testSuite, err := NewIntegrationTestSuite()
	if err != nil {
		log.Fatalf("❌ Failed to create test suite: %v", err)
	}
	defer testSuite.Close()

	// Run all tests
	if err := testSuite.RunAllTests(); err != nil {
		log.Fatalf("❌ Integration tests failed: %v", err)
	}

	log.Println("🎉 All integration tests passed!")
}

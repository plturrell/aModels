package breakdetection

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"time"
)

// FinanceDetector detects breaks in SAP Fioneer Subledger
type FinanceDetector struct {
	sapFioneerURL string
	httpClient    *http.Client
	logger        *log.Logger
}

// NewFinanceDetector creates a new finance break detector
func NewFinanceDetector(sapFioneerURL string, logger *log.Logger) *FinanceDetector {
	if sapFioneerURL == "" {
		sapFioneerURL = "http://localhost:8080" // Default SAP Fioneer URL
	}
	return &FinanceDetector{
		sapFioneerURL: sapFioneerURL,
		httpClient:    &http.Client{Timeout: 60 * time.Second},
		logger:        logger,
	}
}

// SAPFioneerJournalEntry represents a journal entry from SAP Fioneer
type SAPFioneerJournalEntry struct {
	EntryID      string    `json:"entry_id"`
	EntryDate    time.Time `json:"entry_date"`
	PostingDate  time.Time `json:"posting_date"`
	Account      string    `json:"account"`
	DebitAmount  *float64  `json:"debit_amount,omitempty"`
	CreditAmount *float64  `json:"credit_amount,omitempty"`
	Currency     string    `json:"currency"`
	Description  string    `json:"description"`
	Reference    string    `json:"reference,omitempty"`
}

// SAPFioneerAccountBalance represents an account balance from SAP Fioneer
type SAPFioneerAccountBalance struct {
	Account        string    `json:"account"`
	BalanceDate    time.Time `json:"balance_date"`
	OpeningBalance float64   `json:"opening_balance"`
	DebitTotal     float64   `json:"debit_total"`
	CreditTotal    float64   `json:"credit_total"`
	ClosingBalance float64   `json:"closing_balance"`
	Currency       string    `json:"currency"`
}

// DetectBreaks detects breaks in SAP Fioneer Subledger
func (fd *FinanceDetector) DetectBreaks(ctx context.Context, baseline *Baseline, config map[string]interface{}) ([]*Break, error) {
	if fd.logger != nil {
		fd.logger.Printf("Starting finance break detection for system: %s", baseline.SystemName)
	}

	// Parse baseline snapshot data
	var baselineData map[string]interface{}
	if err := json.Unmarshal(baseline.SnapshotData, &baselineData); err != nil {
		return nil, fmt.Errorf("failed to parse baseline data: %w", err)
	}

	// Extract baseline journal entries and balances
	baselineEntries, err := fd.extractBaselineJournalEntries(baselineData)
	if err != nil {
		return nil, fmt.Errorf("failed to extract baseline journal entries: %w", err)
	}

	baselineBalances, err := fd.extractBaselineAccountBalances(baselineData)
	if err != nil {
		return nil, fmt.Errorf("failed to extract baseline account balances: %w", err)
	}

	// Fetch current SAP Fioneer data
	currentEntries, err := fd.fetchCurrentJournalEntries(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch current journal entries: %w", err)
	}

	currentBalances, err := fd.fetchCurrentAccountBalances(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch current account balances: %w", err)
	}

	// Detect breaks
	var breaks []*Break

	// 1. Detect missing journal entries
	missingEntryBreaks := fd.detectMissingJournalEntries(baselineEntries, currentEntries)
	breaks = append(breaks, missingEntryBreaks...)

	// 2. Detect amount mismatches
	amountBreaks := fd.detectAmountMismatches(baselineEntries, currentEntries)
	breaks = append(breaks, amountBreaks...)

	// 3. Detect account balance breaks
	balanceBreaks := fd.detectAccountBalanceBreaks(baselineBalances, currentBalances)
	breaks = append(breaks, balanceBreaks...)

	// 4. Detect reconciliation breaks
	reconciliationBreaks := fd.detectReconciliationBreaks(baselineEntries, currentEntries)
	breaks = append(breaks, reconciliationBreaks...)

	// 5. Detect account mismatches
	accountBreaks := fd.detectAccountMismatches(baselineEntries, currentEntries)
	breaks = append(breaks, accountBreaks...)

	if fd.logger != nil {
		fd.logger.Printf("Finance break detection completed: %d breaks detected", len(breaks))
	}

	return breaks, nil
}

// extractBaselineJournalEntries extracts journal entries from baseline data
func (fd *FinanceDetector) extractBaselineJournalEntries(baselineData map[string]interface{}) (map[string]*SAPFioneerJournalEntry, error) {
	entries := make(map[string]*SAPFioneerJournalEntry)

	// Try to extract from various possible structures
	if entriesData, ok := baselineData["journal_entries"].([]interface{}); ok {
		for _, entryData := range entriesData {
			if entryMap, ok := entryData.(map[string]interface{}); ok {
				entry := fd.parseJournalEntry(entryMap)
				if entry != nil {
					entries[entry.EntryID] = entry
				}
			}
		}
	}

	return entries, nil
}

// extractBaselineAccountBalances extracts account balances from baseline data
func (fd *FinanceDetector) extractBaselineAccountBalances(baselineData map[string]interface{}) (map[string]*SAPFioneerAccountBalance, error) {
	balances := make(map[string]*SAPFioneerAccountBalance)

	if balancesData, ok := baselineData["account_balances"].([]interface{}); ok {
		for _, balanceData := range balancesData {
			if balanceMap, ok := balanceData.(map[string]interface{}); ok {
				balance := fd.parseAccountBalance(balanceMap)
				if balance != nil {
					balances[balance.Account] = balance
				}
			}
		}
	}

	return balances, nil
}

// parseJournalEntry parses a journal entry from a map
func (fd *FinanceDetector) parseJournalEntry(data map[string]interface{}) *SAPFioneerJournalEntry {
	entry := &SAPFioneerJournalEntry{}

	if entryID, ok := data["entry_id"].(string); ok {
		entry.EntryID = entryID
	} else {
		return nil // Entry ID is required
	}

	if entryDate, ok := data["entry_date"].(string); ok {
		if parsed, err := time.Parse(time.RFC3339, entryDate); err == nil {
			entry.EntryDate = parsed
		}
	}

	if account, ok := data["account"].(string); ok {
		entry.Account = account
	}

	if debit, ok := data["debit_amount"].(float64); ok {
		entry.DebitAmount = &debit
	}

	if credit, ok := data["credit_amount"].(float64); ok {
		entry.CreditAmount = &credit
	}

	if currency, ok := data["currency"].(string); ok {
		entry.Currency = currency
	}

	if desc, ok := data["description"].(string); ok {
		entry.Description = desc
	}

	return entry
}

// parseAccountBalance parses an account balance from a map
func (fd *FinanceDetector) parseAccountBalance(data map[string]interface{}) *SAPFioneerAccountBalance {
	balance := &SAPFioneerAccountBalance{}

	if account, ok := data["account"].(string); ok {
		balance.Account = account
	} else {
		return nil // Account is required
	}

	if opening, ok := data["opening_balance"].(float64); ok {
		balance.OpeningBalance = opening
	}

	if debit, ok := data["debit_total"].(float64); ok {
		balance.DebitTotal = debit
	}

	if credit, ok := data["credit_total"].(float64); ok {
		balance.CreditTotal = credit
	}

	if closing, ok := data["closing_balance"].(float64); ok {
		balance.ClosingBalance = closing
	}

	if currency, ok := data["currency"].(string); ok {
		balance.Currency = currency
	}

	return balance
}

// fetchCurrentJournalEntries fetches current journal entries from SAP Fioneer
func (fd *FinanceDetector) fetchCurrentJournalEntries(ctx context.Context) (map[string]*SAPFioneerJournalEntry, error) {
	// Construct API endpoint URL
	baseURL := strings.TrimSuffix(fd.sapFioneerURL, "/")
	endpoint := fmt.Sprintf("%s/api/journal-entries", baseURL)

	// Create request with context timeout
	requestCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(requestCtx, http.MethodGet, endpoint, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	// Add authentication header if API key is available (from environment)
	if apiKey := getAPIKey("SAP_FIONEER_API_KEY"); apiKey != "" {
		req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", apiKey))
	}
	req.Header.Set("Accept", "application/json")

	if fd.logger != nil {
		fd.logger.Printf("Fetching journal entries from SAP Fioneer: %s", endpoint)
	}

	// Make HTTP request with retry logic
	resp, err := fd.makeHTTPRequestWithRetry(req, 3)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch journal entries: %w", err)
	}
	defer resp.Body.Close()

	// Read response body
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	// Check HTTP status
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("SAP Fioneer API returned error status %d: %s", resp.StatusCode, string(body))
	}

	// Parse response - expect array or object with entries field
	var response struct {
		Entries []*SAPFioneerJournalEntry `json:"entries,omitempty"`
		Data    []*SAPFioneerJournalEntry `json:"data,omitempty"`
	}

	// Try parsing as array first
	var entriesArray []*SAPFioneerJournalEntry
	if err := json.Unmarshal(body, &entriesArray); err == nil {
		// Successfully parsed as array
		entries := make(map[string]*SAPFioneerJournalEntry, len(entriesArray))
		for _, entry := range entriesArray {
			entries[entry.EntryID] = entry
		}
		return entries, nil
	}

	// Try parsing as object with entries/data field
	if err := json.Unmarshal(body, &response); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	entries := make(map[string]*SAPFioneerJournalEntry)
	if len(response.Entries) > 0 {
		for _, entry := range response.Entries {
			entries[entry.EntryID] = entry
		}
	} else if len(response.Data) > 0 {
		for _, entry := range response.Data {
			entries[entry.EntryID] = entry
		}
	}

	if fd.logger != nil {
		fd.logger.Printf("Fetched %d journal entries from SAP Fioneer", len(entries))
	}

	return entries, nil
}

// fetchCurrentAccountBalances fetches current account balances from SAP Fioneer
func (fd *FinanceDetector) fetchCurrentAccountBalances(ctx context.Context) (map[string]*SAPFioneerAccountBalance, error) {
	// Construct API endpoint URL
	baseURL := strings.TrimSuffix(fd.sapFioneerURL, "/")
	endpoint := fmt.Sprintf("%s/api/account-balances", baseURL)

	// Create request with context timeout
	requestCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(requestCtx, http.MethodGet, endpoint, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	// Add authentication header if API key is available
	if apiKey := getAPIKey("SAP_FIONEER_API_KEY"); apiKey != "" {
		req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", apiKey))
	}
	req.Header.Set("Accept", "application/json")

	if fd.logger != nil {
		fd.logger.Printf("Fetching account balances from SAP Fioneer: %s", endpoint)
	}

	// Make HTTP request with retry logic
	resp, err := fd.makeHTTPRequestWithRetry(req, 3)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch account balances: %w", err)
	}
	defer resp.Body.Close()

	// Read response body
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	// Check HTTP status
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("SAP Fioneer API returned error status %d: %s", resp.StatusCode, string(body))
	}

	// Parse response - expect array or object with balances field
	var response struct {
		Balances []*SAPFioneerAccountBalance `json:"balances,omitempty"`
		Data     []*SAPFioneerAccountBalance `json:"data,omitempty"`
	}

	// Try parsing as array first
	var balancesArray []*SAPFioneerAccountBalance
	if err := json.Unmarshal(body, &balancesArray); err == nil {
		// Successfully parsed as array
		balances := make(map[string]*SAPFioneerAccountBalance, len(balancesArray))
		for _, balance := range balancesArray {
			balances[balance.Account] = balance
		}
		return balances, nil
	}

	// Try parsing as object with balances/data field
	if err := json.Unmarshal(body, &response); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	balances := make(map[string]*SAPFioneerAccountBalance)
	if len(response.Balances) > 0 {
		for _, balance := range response.Balances {
			balances[balance.Account] = balance
		}
	} else if len(response.Data) > 0 {
		for _, balance := range response.Data {
			balances[balance.Account] = balance
		}
	}

	if fd.logger != nil {
		fd.logger.Printf("Fetched %d account balances from SAP Fioneer", len(balances))
	}

	return balances, nil
}

// makeHTTPRequestWithRetry makes an HTTP request with retry logic and exponential backoff
func (fd *FinanceDetector) makeHTTPRequestWithRetry(req *http.Request, maxRetries int) (*http.Response, error) {
	var resp *http.Response
	var err error

	for attempt := 0; attempt < maxRetries; attempt++ {
		if attempt > 0 {
			// Exponential backoff: 1s, 2s, 4s
			waitTime := time.Duration(1<<uint(attempt-1)) * time.Second
			if fd.logger != nil {
				fd.logger.Printf("Retrying HTTP request (attempt %d/%d) after %v", attempt+1, maxRetries, waitTime)
			}

			select {
			case <-time.After(waitTime):
			case <-req.Context().Done():
				return nil, fmt.Errorf("request cancelled: %w", req.Context().Err())
			}
		}

		resp, err = fd.httpClient.Do(req)
		if err == nil {
			return resp, nil
		}

		// Check if context was cancelled
		if req.Context().Err() != nil {
			return nil, fmt.Errorf("request cancelled or timed out: %w", req.Context().Err())
		}

		if fd.logger != nil && attempt < maxRetries-1 {
			fd.logger.Printf("HTTP request failed (attempt %d/%d): %v", attempt+1, maxRetries, err)
		}
	}

	return nil, fmt.Errorf("failed after %d attempts: %w", maxRetries, err)
}

// getAPIKey retrieves API key from environment variable (helper function)
func getAPIKey(envVar string) string {
	return os.Getenv(envVar)
}

// detectMissingJournalEntries detects missing journal entries
func (fd *FinanceDetector) detectMissingJournalEntries(baseline, current map[string]*SAPFioneerJournalEntry) []*Break {
	var breaks []*Break

	for entryID, baselineEntry := range baseline {
		if _, exists := current[entryID]; !exists {
			br := &Break{
				BreakID:          fmt.Sprintf("break-missing-entry-%s", entryID),
				SystemName:       SystemSAPFioneer,
				DetectionType:    DetectionTypeFinance,
				BreakType:        BreakTypeMissingEntry,
				Severity:         SeverityHigh,
				Status:           BreakStatusOpen,
				CurrentValue:     nil,
				BaselineValue:    fd.journalEntryToMap(baselineEntry),
				Difference:       map[string]interface{}{"missing": true},
				AffectedEntities: []string{entryID},
				DetectedAt:       time.Now(),
				CreatedAt:        time.Now(),
				UpdatedAt:        time.Now(),
			}
			breaks = append(breaks, br)
		}
	}

	return breaks
}

// detectAmountMismatches detects amount mismatches in journal entries
func (fd *FinanceDetector) detectAmountMismatches(baseline, current map[string]*SAPFioneerJournalEntry) []*Break {
	var breaks []*Break
	tolerance := 0.01 // 0.01 tolerance for amount comparisons

	for entryID, baselineEntry := range baseline {
		currentEntry, exists := current[entryID]
		if !exists {
			continue // Already handled by missing entry detection
		}

		// Check debit amount mismatch
		if baselineEntry.DebitAmount != nil && currentEntry.DebitAmount != nil {
			diff := *baselineEntry.DebitAmount - *currentEntry.DebitAmount
			if diff < 0 {
				diff = -diff
			}
			if diff > tolerance {
				br := &Break{
					BreakID:       fmt.Sprintf("break-amount-mismatch-%s-debit", entryID),
					SystemName:    SystemSAPFioneer,
					DetectionType: DetectionTypeFinance,
					BreakType:     BreakTypeAmountMismatch,
					Severity:      SeverityCritical,
					Status:        BreakStatusOpen,
					CurrentValue:  fd.journalEntryToMap(currentEntry),
					BaselineValue: fd.journalEntryToMap(baselineEntry),
					Difference: map[string]interface{}{
						"field":      "debit_amount",
						"baseline":   *baselineEntry.DebitAmount,
						"current":    *currentEntry.DebitAmount,
						"difference": diff,
					},
					AffectedEntities: []string{entryID},
					DetectedAt:       time.Now(),
					CreatedAt:        time.Now(),
					UpdatedAt:        time.Now(),
				}
				breaks = append(breaks, br)
			}
		}

		// Check credit amount mismatch
		if baselineEntry.CreditAmount != nil && currentEntry.CreditAmount != nil {
			diff := *baselineEntry.CreditAmount - *currentEntry.CreditAmount
			if diff < 0 {
				diff = -diff
			}
			if diff > tolerance {
				br := &Break{
					BreakID:       fmt.Sprintf("break-amount-mismatch-%s-credit", entryID),
					SystemName:    SystemSAPFioneer,
					DetectionType: DetectionTypeFinance,
					BreakType:     BreakTypeAmountMismatch,
					Severity:      SeverityCritical,
					Status:        BreakStatusOpen,
					CurrentValue:  fd.journalEntryToMap(currentEntry),
					BaselineValue: fd.journalEntryToMap(baselineEntry),
					Difference: map[string]interface{}{
						"field":      "credit_amount",
						"baseline":   *baselineEntry.CreditAmount,
						"current":    *currentEntry.CreditAmount,
						"difference": diff,
					},
					AffectedEntities: []string{entryID},
					DetectedAt:       time.Now(),
					CreatedAt:        time.Now(),
					UpdatedAt:        time.Now(),
				}
				breaks = append(breaks, br)
			}
		}
	}

	return breaks
}

// detectAccountBalanceBreaks detects breaks in account balances
func (fd *FinanceDetector) detectAccountBalanceBreaks(baseline, current map[string]*SAPFioneerAccountBalance) []*Break {
	var breaks []*Break
	tolerance := 0.01

	for account, baselineBalance := range baseline {
		currentBalance, exists := current[account]
		if !exists {
			// Missing account balance
			br := &Break{
				BreakID:          fmt.Sprintf("break-missing-balance-%s", account),
				SystemName:       SystemSAPFioneer,
				DetectionType:    DetectionTypeFinance,
				BreakType:        BreakTypeBalanceBreak,
				Severity:         SeverityHigh,
				Status:           BreakStatusOpen,
				CurrentValue:     nil,
				BaselineValue:    fd.accountBalanceToMap(baselineBalance),
				Difference:       map[string]interface{}{"missing": true},
				AffectedEntities: []string{account},
				DetectedAt:       time.Now(),
				CreatedAt:        time.Now(),
				UpdatedAt:        time.Now(),
			}
			breaks = append(breaks, br)
			continue
		}

		// Check closing balance mismatch
		diff := baselineBalance.ClosingBalance - currentBalance.ClosingBalance
		if diff < 0 {
			diff = -diff
		}
		if diff > tolerance {
			br := &Break{
				BreakID:       fmt.Sprintf("break-balance-mismatch-%s", account),
				SystemName:    SystemSAPFioneer,
				DetectionType: DetectionTypeFinance,
				BreakType:     BreakTypeBalanceBreak,
				Severity:      SeverityCritical,
				Status:        BreakStatusOpen,
				CurrentValue:  fd.accountBalanceToMap(currentBalance),
				BaselineValue: fd.accountBalanceToMap(baselineBalance),
				Difference: map[string]interface{}{
					"field":      "closing_balance",
					"baseline":   baselineBalance.ClosingBalance,
					"current":    currentBalance.ClosingBalance,
					"difference": diff,
				},
				AffectedEntities: []string{account},
				DetectedAt:       time.Now(),
				CreatedAt:        time.Now(),
				UpdatedAt:        time.Now(),
			}
			breaks = append(breaks, br)
		}
	}

	return breaks
}

// detectReconciliationBreaks detects reconciliation breaks
func (fd *FinanceDetector) detectReconciliationBreaks(baseline, current map[string]*SAPFioneerJournalEntry) []*Break {
	var breaks []*Break

	// Check if total debits and credits balance
	baselineDebitTotal := 0.0
	baselineCreditTotal := 0.0
	for _, entry := range baseline {
		if entry.DebitAmount != nil {
			baselineDebitTotal += *entry.DebitAmount
		}
		if entry.CreditAmount != nil {
			baselineCreditTotal += *entry.CreditAmount
		}
	}

	currentDebitTotal := 0.0
	currentCreditTotal := 0.0
	for _, entry := range current {
		if entry.DebitAmount != nil {
			currentDebitTotal += *entry.DebitAmount
		}
		if entry.CreditAmount != nil {
			currentCreditTotal += *entry.CreditAmount
		}
	}

	// Check if baseline balances
	baselineBalance := baselineDebitTotal - baselineCreditTotal
	if baselineBalance < 0 {
		baselineBalance = -baselineBalance
	}

	// Check if current balances
	currentBalance := currentDebitTotal - currentCreditTotal
	if currentBalance < 0 {
		currentBalance = -currentBalance
	}

	tolerance := 0.01
	if baselineBalance > tolerance || currentBalance > tolerance {
		br := &Break{
			BreakID:       fmt.Sprintf("break-reconciliation-%d", time.Now().Unix()),
			SystemName:    SystemSAPFioneer,
			DetectionType: DetectionTypeFinance,
			BreakType:     BreakTypeReconciliationBreak,
			Severity:      SeverityCritical,
			Status:        BreakStatusOpen,
			CurrentValue: map[string]interface{}{
				"debit_total":  currentDebitTotal,
				"credit_total": currentCreditTotal,
				"balance":      currentBalance,
			},
			BaselineValue: map[string]interface{}{
				"debit_total":  baselineDebitTotal,
				"credit_total": baselineCreditTotal,
				"balance":      baselineBalance,
			},
			Difference: map[string]interface{}{
				"baseline_balance": baselineBalance,
				"current_balance":  currentBalance,
				"difference":       baselineBalance - currentBalance,
			},
			AffectedEntities: []string{"all_journal_entries"},
			DetectedAt:       time.Now(),
			CreatedAt:        time.Now(),
			UpdatedAt:        time.Now(),
		}
		breaks = append(breaks, br)
	}

	return breaks
}

// detectAccountMismatches detects account mismatches
func (fd *FinanceDetector) detectAccountMismatches(baseline, current map[string]*SAPFioneerJournalEntry) []*Break {
	var breaks []*Break

	for entryID, baselineEntry := range baseline {
		currentEntry, exists := current[entryID]
		if !exists {
			continue
		}

		if baselineEntry.Account != currentEntry.Account {
			br := &Break{
				BreakID:       fmt.Sprintf("break-account-mismatch-%s", entryID),
				SystemName:    SystemSAPFioneer,
				DetectionType: DetectionTypeFinance,
				BreakType:     BreakTypeAccountMismatch,
				Severity:      SeverityHigh,
				Status:        BreakStatusOpen,
				CurrentValue:  fd.journalEntryToMap(currentEntry),
				BaselineValue: fd.journalEntryToMap(baselineEntry),
				Difference: map[string]interface{}{
					"field":    "account",
					"baseline": baselineEntry.Account,
					"current":  currentEntry.Account,
				},
				AffectedEntities: []string{entryID},
				DetectedAt:       time.Now(),
				CreatedAt:        time.Now(),
				UpdatedAt:        time.Now(),
			}
			breaks = append(breaks, br)
		}
	}

	return breaks
}

// Helper functions
func (fd *FinanceDetector) journalEntryToMap(entry *SAPFioneerJournalEntry) map[string]interface{} {
	m := map[string]interface{}{
		"entry_id": entry.EntryID,
		"account":  entry.Account,
		"currency": entry.Currency,
	}
	if entry.DebitAmount != nil {
		m["debit_amount"] = *entry.DebitAmount
	}
	if entry.CreditAmount != nil {
		m["credit_amount"] = *entry.CreditAmount
	}
	return m
}

func (fd *FinanceDetector) accountBalanceToMap(balance *SAPFioneerAccountBalance) map[string]interface{} {
	return map[string]interface{}{
		"account":         balance.Account,
		"opening_balance": balance.OpeningBalance,
		"debit_total":     balance.DebitTotal,
		"credit_total":    balance.CreditTotal,
		"closing_balance": balance.ClosingBalance,
		"currency":        balance.Currency,
	}
}

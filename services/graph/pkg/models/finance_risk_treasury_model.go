package models

import (
	"context"
	"fmt"
	"time"
)

// FinanceRiskTreasuryModel defines domain-specific node types and relationships
// for Finance, Risk, and Treasury operations.
type FinanceRiskTreasuryModel struct{}

// Node types for Finance-Risk-Treasury domain
const (
	NodeTypeTrade                = "Trade"
	NodeTypeJournalEntry         = "JournalEntry"
	NodeTypeRegulatoryCalculation = "RegulatoryCalculation"
	NodeTypeReportingRule        = "ReportingRule"
	NodeTypeCounterparty         = "Counterparty"
	NodeTypePortfolio            = "Portfolio"
	NodeTypeInstrument           = "Instrument"
	NodeTypeRiskFactor           = "RiskFactor"
)

// Edge types for Finance-Risk-Treasury domain
const (
	EdgeTypeTradesTo             = "TRADES_TO"
	EdgeTypeGeneratesJournalEntry = "GENERATES_JOURNAL_ENTRY"
	EdgeTypeRequiresCalculation   = "REQUIRES_CALCULATION"
	EdgeTypeSatisfiesRule         = "SATISFIES_RULE"
	EdgeTypeBelongsToPortfolio    = "BELONGS_TO_PORTFOLIO"
	EdgeTypeHasCounterparty       = "HAS_COUNTERPARTY"
	EdgeTypeUsesInstrument        = "USES_INSTRUMENT"
	EdgeTypeCalculatesRisk        = "CALCULATES_RISK"
	EdgeTypeFeedsToReport         = "FEEDS_TO_REPORT"
)

// Trade represents a financial trade node in the knowledge graph.
type Trade struct {
	ID              string                 `json:"id"`
	TradeID         string                 `json:"trade_id"`
	TradeDate       time.Time              `json:"trade_date"`
	ValueDate       time.Time              `json:"value_date"`
	MaturityDate    *time.Time             `json:"maturity_date,omitempty"`
	TradeType       string                 `json:"trade_type"` // "FX", "Bond", "Swap", "Option", etc.
	NotionalAmount  float64                `json:"notional_amount"`
	Currency        string                 `json:"currency"`
	Status          string                 `json:"status"` // "Pending", "Executed", "Settled", "Cancelled"
	SourceSystem    string                 `json:"source_system"` // "Murex", "SAP", etc.
	Book            string                 `json:"book,omitempty"`
	Desk            string                 `json:"desk,omitempty"`
	Properties      map[string]interface{} `json:"properties,omitempty"`
}

// ToGraphNode converts a Trade to a knowledge graph Node.
func (t *Trade) ToGraphNode() *DomainNode {
	props := make(map[string]interface{})
	props["trade_id"] = t.TradeID
	props["trade_date"] = t.TradeDate.Format(time.RFC3339)
	props["value_date"] = t.ValueDate.Format(time.RFC3339)
	if t.MaturityDate != nil {
		props["maturity_date"] = t.MaturityDate.Format(time.RFC3339)
	}
	props["trade_type"] = t.TradeType
	props["notional_amount"] = t.NotionalAmount
	props["currency"] = t.Currency
	props["status"] = t.Status
	props["source_system"] = t.SourceSystem
	if t.Book != "" {
		props["book"] = t.Book
	}
	if t.Desk != "" {
		props["desk"] = t.Desk
	}
	// Merge additional properties
	for k, v := range t.Properties {
		props[k] = v
	}

	return &DomainNode{
		ID:         t.ID,
		Type:       NodeTypeTrade,
		Label:      fmt.Sprintf("Trade %s", t.TradeID),
		Properties: props,
	}
}

// JournalEntry represents a journal entry node in the knowledge graph.
type JournalEntry struct {
	ID            string                 `json:"id"`
	EntryID       string                 `json:"entry_id"`
	EntryDate     time.Time              `json:"entry_date"`
	PostingDate   time.Time              `json:"posting_date"`
	Account       string                 `json:"account"`
	DebitAmount   *float64               `json:"debit_amount,omitempty"`
	CreditAmount  *float64               `json:"credit_amount,omitempty"`
	Currency      string                 `json:"currency"`
	Description   string                 `json:"description"`
	SourceSystem  string                 `json:"source_system"` // "SAP_GL", "Murex", etc.
	DocumentType  string                 `json:"document_type,omitempty"`
	Properties    map[string]interface{} `json:"properties,omitempty"`
}

// ToGraphNode converts a JournalEntry to a knowledge graph Node.
func (je *JournalEntry) ToGraphNode() *DomainNode {
	props := make(map[string]interface{})
	props["entry_id"] = je.EntryID
	props["entry_date"] = je.EntryDate.Format(time.RFC3339)
	props["posting_date"] = je.PostingDate.Format(time.RFC3339)
	props["account"] = je.Account
	if je.DebitAmount != nil {
		props["debit_amount"] = *je.DebitAmount
	}
	if je.CreditAmount != nil {
		props["credit_amount"] = *je.CreditAmount
	}
	props["currency"] = je.Currency
	props["description"] = je.Description
	props["source_system"] = je.SourceSystem
	if je.DocumentType != "" {
		props["document_type"] = je.DocumentType
	}
	// Merge additional properties
	for k, v := range je.Properties {
		props[k] = v
	}

	return &DomainNode{
		ID:         je.ID,
		Type:       NodeTypeJournalEntry,
		Label:      fmt.Sprintf("Journal Entry %s", je.EntryID),
		Properties: props,
	}
}

// RegulatoryCalculation represents a regulatory calculation node.
type RegulatoryCalculation struct {
	ID                string                 `json:"id"`
	CalculationID     string                 `json:"calculation_id"`
	CalculationType   string                 `json:"calculation_type"` // "Capital", "Liquidity", "Risk", "MAS610", "BCBS239"
	CalculationDate   time.Time              `json:"calculation_date"`
	ReportPeriod      string                 `json:"report_period"` // "Daily", "Monthly", "Quarterly"
	Result            float64                `json:"result"`
	Currency          string                 `json:"currency"`
	RegulatoryFramework string               `json:"regulatory_framework"` // "MAS 610", "BCBS 239", etc.
	SourceSystem      string                 `json:"source_system"` // "Murex", "FPSL", "BCRS", etc.
	Status            string                 `json:"status"` // "Pending", "Calculated", "Validated", "Submitted"
	Properties        map[string]interface{} `json:"properties,omitempty"`
}

// ToGraphNode converts a RegulatoryCalculation to a knowledge graph Node.
func (rc *RegulatoryCalculation) ToGraphNode() *DomainNode {
	props := make(map[string]interface{})
	props["calculation_id"] = rc.CalculationID
	props["calculation_type"] = rc.CalculationType
	props["calculation_date"] = rc.CalculationDate.Format(time.RFC3339)
	props["report_period"] = rc.ReportPeriod
	props["result"] = rc.Result
	props["currency"] = rc.Currency
	props["regulatory_framework"] = rc.RegulatoryFramework
	props["source_system"] = rc.SourceSystem
	props["status"] = rc.Status
	// Merge additional properties
	for k, v := range rc.Properties {
		props[k] = v
	}

	return &DomainNode{
		ID:         rc.ID,
		Type:       NodeTypeRegulatoryCalculation,
		Label:      fmt.Sprintf("Regulatory Calculation %s", rc.CalculationID),
		Properties: props,
	}
}

// ReportingRule represents a regulatory reporting rule node.
type ReportingRule struct {
	ID                  string                 `json:"id"`
	RuleID              string                 `json:"rule_id"`
	RuleName            string                 `json:"rule_name"`
	RegulatoryFramework string                 `json:"regulatory_framework"` // "MAS 610", "BCBS 239", etc.
	RuleType            string                 `json:"rule_type"` // "FieldDefinition", "Validation", "Calculation", "Aggregation"
	Description         string                 `json:"description"`
	RuleExpression      string                 `json:"rule_expression,omitempty"` // SQL, formula, or expression
	ApplicableTo        []string               `json:"applicable_to,omitempty"` // Report sections, fields, etc.
	EffectiveDate       time.Time              `json:"effective_date"`
	ExpiryDate          *time.Time              `json:"expiry_date,omitempty"`
	Status              string                 `json:"status"` // "Active", "Draft", "Deprecated"
	SourceDocument      string                 `json:"source_document,omitempty"`
	Properties          map[string]interface{} `json:"properties,omitempty"`
}

// ToGraphNode converts a ReportingRule to a knowledge graph Node.
func (rr *ReportingRule) ToGraphNode() *DomainNode {
	props := make(map[string]interface{})
	props["rule_id"] = rr.RuleID
	props["rule_name"] = rr.RuleName
	props["regulatory_framework"] = rr.RegulatoryFramework
	props["rule_type"] = rr.RuleType
	props["description"] = rr.Description
	if rr.RuleExpression != "" {
		props["rule_expression"] = rr.RuleExpression
	}
	if len(rr.ApplicableTo) > 0 {
		props["applicable_to"] = rr.ApplicableTo
	}
	props["effective_date"] = rr.EffectiveDate.Format(time.RFC3339)
	if rr.ExpiryDate != nil {
		props["expiry_date"] = rr.ExpiryDate.Format(time.RFC3339)
	}
	props["status"] = rr.Status
	if rr.SourceDocument != "" {
		props["source_document"] = rr.SourceDocument
	}
	// Merge additional properties
	for k, v := range rr.Properties {
		props[k] = v
	}

	return &DomainNode{
		ID:         rr.ID,
		Type:       NodeTypeReportingRule,
		Label:      fmt.Sprintf("Reporting Rule %s", rr.RuleID),
		Properties: props,
	}
}

// DomainNode represents a domain-specific node in the knowledge graph.
type DomainNode struct {
	ID         string                 `json:"id"`
	Type       string                 `json:"type"`
	Label      string                 `json:"label"`
	Properties map[string]interface{} `json:"properties"`
}

// DomainEdge represents a domain-specific edge in the knowledge graph.
type DomainEdge struct {
	SourceID   string                 `json:"source_id"`
	TargetID   string                 `json:"target_id"`
	Type       string                 `json:"type"`
	Label      string                 `json:"label"`
	Properties map[string]interface{} `json:"properties,omitempty"`
}

// CreateTradeToJournalEntryEdge creates an edge from a Trade to a JournalEntry.
func CreateTradeToJournalEntryEdge(tradeID, journalEntryID string, props map[string]interface{}) *DomainEdge {
	if props == nil {
		props = make(map[string]interface{})
	}
	props["created_at"] = time.Now().Format(time.RFC3339)
	return &DomainEdge{
		SourceID:   tradeID,
		TargetID:   journalEntryID,
		Type:       EdgeTypeGeneratesJournalEntry,
		Label:      "generates journal entry",
		Properties: props,
	}
}

// CreateTradeToCalculationEdge creates an edge from a Trade to a RegulatoryCalculation.
func CreateTradeToCalculationEdge(tradeID, calculationID string, props map[string]interface{}) *DomainEdge {
	if props == nil {
		props = make(map[string]interface{})
	}
	props["created_at"] = time.Now().Format(time.RFC3339)
	return &DomainEdge{
		SourceID:   tradeID,
		TargetID:   calculationID,
		Type:       EdgeTypeRequiresCalculation,
		Label:      "requires calculation",
		Properties: props,
	}
}

// CreateCalculationToRuleEdge creates an edge from a RegulatoryCalculation to a ReportingRule.
func CreateCalculationToRuleEdge(calculationID, ruleID string, props map[string]interface{}) *DomainEdge {
	if props == nil {
		props = make(map[string]interface{})
	}
	props["created_at"] = time.Now().Format(time.RFC3339)
	return &DomainEdge{
		SourceID:   calculationID,
		TargetID:   ruleID,
		Type:       EdgeTypeSatisfiesRule,
		Label:      "satisfies rule",
		Properties: props,
	}
}

// ModelMapper maps source system data to Finance-Risk-Treasury domain nodes.
type ModelMapper interface {
	MapTrade(ctx context.Context, sourceData map[string]interface{}) (*Trade, error)
	MapJournalEntry(ctx context.Context, sourceData map[string]interface{}) (*JournalEntry, error)
	MapRegulatoryCalculation(ctx context.Context, sourceData map[string]interface{}) (*RegulatoryCalculation, error)
	MapReportingRule(ctx context.Context, sourceData map[string]interface{}) (*ReportingRule, error)
}

// DefaultModelMapper provides default mapping implementations.
type DefaultModelMapper struct{}

// NewDefaultModelMapper creates a new DefaultModelMapper.
func NewDefaultModelMapper() *DefaultModelMapper {
	return &DefaultModelMapper{}
}

// MapTrade maps source data to a Trade node.
func (dm *DefaultModelMapper) MapTrade(ctx context.Context, sourceData map[string]interface{}) (*Trade, error) {
	trade := &Trade{
		ID:         fmt.Sprintf("trade-%v", sourceData["trade_id"]),
		Properties: make(map[string]interface{}),
	}

	if tradeID, ok := sourceData["trade_id"].(string); ok {
		trade.TradeID = tradeID
	}
	if tradeDate, ok := sourceData["trade_date"].(string); ok {
		if t, err := time.Parse(time.RFC3339, tradeDate); err == nil {
			trade.TradeDate = t
		}
	}
	if tradeType, ok := sourceData["trade_type"].(string); ok {
		trade.TradeType = tradeType
	}
	if sourceSystem, ok := sourceData["source_system"].(string); ok {
		trade.SourceSystem = sourceSystem
	}

	// Copy remaining properties
	for k, v := range sourceData {
		if k != "trade_id" && k != "trade_date" && k != "trade_type" && k != "source_system" {
			trade.Properties[k] = v
		}
	}

	return trade, nil
}

// MapJournalEntry maps source data to a JournalEntry node.
func (dm *DefaultModelMapper) MapJournalEntry(ctx context.Context, sourceData map[string]interface{}) (*JournalEntry, error) {
	entry := &JournalEntry{
		ID:         fmt.Sprintf("journal-entry-%v", sourceData["entry_id"]),
		Properties: make(map[string]interface{}),
	}

	if entryID, ok := sourceData["entry_id"].(string); ok {
		entry.EntryID = entryID
	}
	if sourceSystem, ok := sourceData["source_system"].(string); ok {
		entry.SourceSystem = sourceSystem
	}

	// Copy remaining properties
	for k, v := range sourceData {
		if k != "entry_id" && k != "source_system" {
			entry.Properties[k] = v
		}
	}

	return entry, nil
}

// MapRegulatoryCalculation maps source data to a RegulatoryCalculation node.
func (dm *DefaultModelMapper) MapRegulatoryCalculation(ctx context.Context, sourceData map[string]interface{}) (*RegulatoryCalculation, error) {
	calc := &RegulatoryCalculation{
		ID:         fmt.Sprintf("regulatory-calc-%v", sourceData["calculation_id"]),
		Properties: make(map[string]interface{}),
	}

	if calcID, ok := sourceData["calculation_id"].(string); ok {
		calc.CalculationID = calcID
	}
	if calcType, ok := sourceData["calculation_type"].(string); ok {
		calc.CalculationType = calcType
	}
	if framework, ok := sourceData["regulatory_framework"].(string); ok {
		calc.RegulatoryFramework = framework
	}
	if sourceSystem, ok := sourceData["source_system"].(string); ok {
		calc.SourceSystem = sourceSystem
	}

	// Copy remaining properties
	for k, v := range sourceData {
		if k != "calculation_id" && k != "calculation_type" && k != "regulatory_framework" && k != "source_system" {
			calc.Properties[k] = v
		}
	}

	return calc, nil
}

// MapReportingRule maps source data to a ReportingRule node.
func (dm *DefaultModelMapper) MapReportingRule(ctx context.Context, sourceData map[string]interface{}) (*ReportingRule, error) {
	rule := &ReportingRule{
		ID:         fmt.Sprintf("reporting-rule-%v", sourceData["rule_id"]),
		Properties: make(map[string]interface{}),
	}

	if ruleID, ok := sourceData["rule_id"].(string); ok {
		rule.RuleID = ruleID
	}
	if ruleName, ok := sourceData["rule_name"].(string); ok {
		rule.RuleName = ruleName
	}
	if framework, ok := sourceData["regulatory_framework"].(string); ok {
		rule.RegulatoryFramework = framework
	}

	// Copy remaining properties
	for k, v := range sourceData {
		if k != "rule_id" && k != "rule_name" && k != "regulatory_framework" {
			rule.Properties[k] = v
		}
	}

	return rule, nil
}


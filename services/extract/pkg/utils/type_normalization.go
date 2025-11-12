package utils

import (
	"regexp"
	"strings"
)

// NormalizeToCanonicalType maps database-specific types to canonical types
// used for quality metrics calculation. This ensures consistency between
// actual distribution and ideal distribution keys.
//
// Canonical types:
// - "string": Text types (VARCHAR, TEXT, CHAR, STRING, etc.)
// - "number": Numeric types (INT, BIGINT, DECIMAL, NUMERIC, FLOAT, DOUBLE, etc.)
// - "boolean": Boolean types (BOOLEAN, BOOL, BIT(1))
// - "date": Date types (DATE)
// - "timestamp": Timestamp types (TIMESTAMP, DATETIME, TIME)
// - "array": Array types (ARRAY)
// - "object": Object types (JSON, OBJECT, STRUCT, MAP)
func NormalizeToCanonicalType(rawType string) string {
	if rawType == "" {
		return "string" // Default to string for empty types
	}

	// Normalize: lowercase, trim whitespace, remove parentheses and size constraints
	normalized := strings.ToLower(strings.TrimSpace(rawType))

	// Remove size constraints and parameters (e.g., "VARCHAR(255)" -> "varchar")
	// Match patterns like (255), (10,2), etc.
	sizePattern := regexp.MustCompile(`\([^)]*\)`)
	normalized = sizePattern.ReplaceAllString(normalized, "")
	normalized = strings.TrimSpace(normalized)

	// Remove common prefixes/suffixes that don't affect type category
	normalized = strings.TrimPrefix(normalized, "unsigned ")
	normalized = strings.TrimPrefix(normalized, "signed ")
	normalized = strings.TrimSuffix(normalized, " unsigned")
	normalized = strings.TrimSuffix(normalized, " signed")

	// Map to canonical types
	// String types
	if isStringType(normalized) {
		return "string"
	}

	// Number types
	if isNumberType(normalized) {
		return "number"
	}

	// Boolean types
	if isBooleanType(normalized) {
		return "boolean"
	}

	// Date types
	if isDateType(normalized) {
		return "date"
	}

	// Timestamp types (map to date for canonical comparison)
	if isTimestampType(normalized) {
		return "date"
	}

	// Array types
	if isArrayType(normalized) {
		return "array"
	}

	// Object types
	if isObjectType(normalized) {
		return "object"
	}

	// Default: unknown types default to string
	return "string"
}

// isStringType checks if a type is a string/text type
func isStringType(t string) bool {
	stringTypes := []string{
		"string", "varchar", "char", "text", "character",
		"nvarchar", "nchar", "ntext", "clob", "blob",
		"longvarchar", "varbinary", "binary",
		"citext", "name", "bpchar", // Postgres specific
		"hstring", "hchar", // HANA specific
	}

	for _, st := range stringTypes {
		if strings.Contains(t, st) {
			return true
		}
	}
	return false
}

// isNumberType checks if a type is a numeric type
func isNumberType(t string) bool {
	numberTypes := []string{
		"int", "integer", "bigint", "smallint", "tinyint",
		"decimal", "numeric", "number", "float", "double",
		"real", "money", "smallmoney", "bit", // SQL Server bit can be numeric
		"serial", "bigserial", "smallserial", // Postgres serial types
		"double precision", "single precision",
		"int2", "int4", "int8", // Postgres integer sizes
		"float4", "float8", // Postgres float sizes
		"decfloat", "decfloat16", "decfloat34", // HANA decimal float
		"tinyint", "smallint", "mediumint", // MySQL integer sizes
	}

	// Special case: bit(1) is boolean, but bit without size or bit(>1) can be numeric
	if t == "bit" {
		return false // Handled by boolean check
	}

	for _, nt := range numberTypes {
		if strings.Contains(t, nt) {
			return true
		}
	}
	return false
}

// isBooleanType checks if a type is a boolean type
func isBooleanType(t string) bool {
	booleanTypes := []string{
		"boolean", "bool",
	}

	// Special case: bit(1) is often used as boolean
	if t == "bit" || strings.HasPrefix(t, "bit(") {
		return true
	}

	for _, bt := range booleanTypes {
		if t == bt {
			return true
		}
	}
	return false
}

// isDateType checks if a type is a date type (without time)
func isDateType(t string) bool {
	dateTypes := []string{
		"date",
	}

	for _, dt := range dateTypes {
		if t == dt {
			return true
		}
	}
	return false
}

// isTimestampType checks if a type is a timestamp/datetime type
func isTimestampType(t string) bool {
	timestampTypes := []string{
		"timestamp", "datetime", "time",
		"timestamptz", "timetz", // Postgres timezone-aware types
		"smalldatetime", "datetime2", "datetimeoffset", // SQL Server
		"year", // MySQL year type
	}

	for _, tt := range timestampTypes {
		if strings.Contains(t, tt) {
			return true
		}
	}
	return false
}

// isArrayType checks if a type is an array type
func isArrayType(t string) bool {
	arrayTypes := []string{
		"array", "list",
	}

	// Postgres arrays are denoted with [] suffix
	if strings.HasSuffix(t, "[]") {
		return true
	}

	for _, at := range arrayTypes {
		if strings.Contains(t, at) {
			return true
		}
	}
	return false
}

// isObjectType checks if a type is an object/structured type
func isObjectType(t string) bool {
	objectTypes := []string{
		"json", "jsonb", "object", "struct", "map",
		"xml", "hstore", // Postgres specific
		"record", "row", // Postgres record types
		"variant", // Snowflake variant
	}

	for _, ot := range objectTypes {
		if strings.Contains(t, ot) {
			return true
		}
	}
	return false
}

// GetCanonicalTypeDistribution calculates the distribution of canonical types
// from a slice of raw type strings. Returns a map of canonical type -> count.
func GetCanonicalTypeDistribution(rawTypes []string) map[string]int {
	distribution := make(map[string]int)

	// Initialize all canonical types to 0
	canonicalTypes := []string{"string", "number", "boolean", "date", "array", "object"}
	for _, ct := range canonicalTypes {
		distribution[ct] = 0
	}

	// Count normalized types
	for _, rawType := range rawTypes {
		canonical := NormalizeToCanonicalType(rawType)
		distribution[canonical]++
	}

	return distribution
}

// GetCanonicalTypeDistributionNormalized returns the normalized distribution
// (probabilities) of canonical types from raw type strings.
func GetCanonicalTypeDistributionNormalized(rawTypes []string) map[string]float64 {
	distribution := GetCanonicalTypeDistribution(rawTypes)
	total := float64(len(rawTypes))

	if total == 0 {
		return make(map[string]float64)
	}

	normalized := make(map[string]float64)
	for canonical, count := range distribution {
		normalized[canonical] = float64(count) / total
	}

	return normalized
}

/**
 * Input Validation Utilities
 * Apple-standard security-first validation
 */

export interface ValidationResult {
  isValid: boolean;
  error?: string;
}

/**
 * Sanitize string input to prevent XSS
 */
export function sanitizeString(input: string): string {
  if (typeof input !== 'string') return '';
  
  return input
    .trim()
    .replace(/[<>]/g, '') // Remove angle brackets
    .replace(/javascript:/gi, '') // Remove javascript: protocol
    .replace(/on\w+=/gi, ''); // Remove inline event handlers
}

/**
 * Validate URL with strict protocol checks
 */
export function validateUrl(url: string): ValidationResult {
  if (!url || typeof url !== 'string') {
    return { isValid: false, error: 'URL is required' };
  }

  try {
    const parsed = new URL(url);
    
    // Only allow safe protocols
    const allowedProtocols = ['http:', 'https:', 'file:'];
    if (!allowedProtocols.includes(parsed.protocol)) {
      return {
        isValid: false,
        error: `Protocol ${parsed.protocol} is not allowed. Use http:, https:, or file:`,
      };
    }

    return { isValid: true };
  } catch (error) {
    return { isValid: false, error: 'Invalid URL format' };
  }
}

/**
 * Validate request ID format
 */
export function validateRequestId(id: string): ValidationResult {
  if (!id || typeof id !== 'string') {
    return { isValid: false, error: 'Request ID is required' };
  }

  // Allow alphanumeric, hyphens, underscores
  const idPattern = /^[a-zA-Z0-9_-]+$/;
  if (!idPattern.test(id)) {
    return {
      isValid: false,
      error: 'Request ID can only contain letters, numbers, hyphens, and underscores',
    };
  }

  if (id.length > 128) {
    return { isValid: false, error: 'Request ID is too long (max 128 characters)' };
  }

  return { isValid: true };
}

/**
 * Validate table name
 */
export function validateTableName(name: string): ValidationResult {
  if (!name || typeof name !== 'string') {
    return { isValid: false, error: 'Table name is required' };
  }

  // SQL-safe table names
  const tablePattern = /^[a-zA-Z_][a-zA-Z0-9_]*$/;
  if (!tablePattern.test(name)) {
    return {
      isValid: false,
      error: 'Table name must start with a letter and contain only letters, numbers, and underscores',
    };
  }

  if (name.length > 64) {
    return { isValid: false, error: 'Table name is too long (max 64 characters)' };
  }

  return { isValid: true };
}

/**
 * Validate query string with length limit
 */
export function validateQuery(query: string, maxLength = 10000): ValidationResult {
  if (!query || typeof query !== 'string') {
    return { isValid: false, error: 'Query is required' };
  }

  const trimmed = query.trim();
  if (trimmed.length === 0) {
    return { isValid: false, error: 'Query cannot be empty' };
  }

  if (trimmed.length > maxLength) {
    return {
      isValid: false,
      error: `Query is too long (max ${maxLength} characters)`,
    };
  }

  return { isValid: true };
}

/**
 * Validate positive integer
 */
export function validatePositiveInteger(value: any, fieldName: string): ValidationResult {
  if (value === undefined || value === null) {
    return { isValid: false, error: `${fieldName} is required` };
  }

  const num = Number(value);
  if (!Number.isInteger(num) || num <= 0) {
    return { isValid: false, error: `${fieldName} must be a positive integer` };
  }

  return { isValid: true };
}

/**
 * Validate email format
 */
export function validateEmail(email: string): ValidationResult {
  if (!email || typeof email !== 'string') {
    return { isValid: false, error: 'Email is required' };
  }

  const emailPattern = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  if (!emailPattern.test(email)) {
    return { isValid: false, error: 'Invalid email format' };
  }

  return { isValid: true };
}

/**
 * Validate JSON string
 */
export function validateJSON(jsonString: string): ValidationResult {
  if (!jsonString || typeof jsonString !== 'string') {
    return { isValid: false, error: 'JSON string is required' };
  }

  try {
    JSON.parse(jsonString);
    return { isValid: true };
  } catch (error) {
    return { isValid: false, error: 'Invalid JSON format' };
  }
}

/**
 * Rate limiter for preventing abuse
 */
export class RateLimiter {
  private requests: Map<string, number[]> = new Map();
  private readonly maxRequests: number;
  private readonly windowMs: number;

  constructor(maxRequests: number = 10, windowMs: number = 60000) {
    this.maxRequests = maxRequests;
    this.windowMs = windowMs;
  }

  /**
   * Check if request is allowed
   */
  isAllowed(key: string): boolean {
    const now = Date.now();
    const timestamps = this.requests.get(key) || [];

    // Remove old timestamps outside the window
    const validTimestamps = timestamps.filter((ts) => now - ts < this.windowMs);

    if (validTimestamps.length >= this.maxRequests) {
      return false;
    }

    // Add current timestamp
    validTimestamps.push(now);
    this.requests.set(key, validTimestamps);

    return true;
  }

  /**
   * Reset rate limit for a key
   */
  reset(key: string): void {
    this.requests.delete(key);
  }

  /**
   * Clear all rate limits
   */
  clearAll(): void {
    this.requests.clear();
  }
}

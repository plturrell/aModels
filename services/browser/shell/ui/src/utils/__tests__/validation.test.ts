import { describe, it, expect } from 'vitest';
import {
  sanitizeString,
  validateUrl,
  validateRequestId,
  validateTableName,
  validateQuery,
  validatePositiveInteger,
  validateEmail,
  validateJSON,
  RateLimiter,
} from '../validation';

describe('sanitizeString', () => {
  it('should remove angle brackets', () => {
    expect(sanitizeString('<script>alert("xss")</script>')).toBe('scriptalert("xss")/script');
  });

  it('should remove javascript: protocol', () => {
    expect(sanitizeString('javascript:alert(1)')).toBe('alert(1)');
  });

  it('should remove inline event handlers', () => {
    expect(sanitizeString('onclick=alert(1)')).toBe('alert(1)');
  });

  it('should trim whitespace', () => {
    expect(sanitizeString('  hello  ')).toBe('hello');
  });
});

describe('validateUrl', () => {
  it('should accept valid http URLs', () => {
    const result = validateUrl('http://localhost:8000');
    expect(result.isValid).toBe(true);
  });

  it('should accept valid https URLs', () => {
    const result = validateUrl('https://example.com');
    expect(result.isValid).toBe(true);
  });

  it('should reject javascript: protocol', () => {
    const result = validateUrl('javascript:alert(1)');
    expect(result.isValid).toBe(false);
    expect(result.error).toContain('not allowed');
  });

  it('should reject invalid URLs', () => {
    const result = validateUrl('not a url');
    expect(result.isValid).toBe(false);
  });
});

describe('validateRequestId', () => {
  it('should accept valid request IDs', () => {
    const result = validateRequestId('req-123_abc');
    expect(result.isValid).toBe(true);
  });

  it('should reject IDs with special characters', () => {
    const result = validateRequestId('req@123');
    expect(result.isValid).toBe(false);
  });

  it('should reject too-long IDs', () => {
    const result = validateRequestId('a'.repeat(129));
    expect(result.isValid).toBe(false);
    expect(result.error).toContain('too long');
  });
});

describe('validateTableName', () => {
  it('should accept valid table names', () => {
    const result = validateTableName('users_table');
    expect(result.isValid).toBe(true);
  });

  it('should reject names starting with numbers', () => {
    const result = validateTableName('123_table');
    expect(result.isValid).toBe(false);
  });

  it('should reject names with special characters', () => {
    const result = validateTableName('users-table');
    expect(result.isValid).toBe(false);
  });
});

describe('validateQuery', () => {
  it('should accept valid queries', () => {
    const result = validateQuery('SELECT * FROM users');
    expect(result.isValid).toBe(true);
  });

  it('should reject empty queries', () => {
    const result = validateQuery('   ');
    expect(result.isValid).toBe(false);
  });

  it('should reject too-long queries', () => {
    const result = validateQuery('a'.repeat(10001));
    expect(result.isValid).toBe(false);
    expect(result.error).toContain('too long');
  });
});

describe('validateEmail', () => {
  it('should accept valid emails', () => {
    const result = validateEmail('user@example.com');
    expect(result.isValid).toBe(true);
  });

  it('should reject invalid emails', () => {
    const result = validateEmail('not-an-email');
    expect(result.isValid).toBe(false);
  });
});

describe('RateLimiter', () => {
  it('should allow requests within limit', () => {
    const limiter = new RateLimiter(3, 1000);
    expect(limiter.isAllowed('user1')).toBe(true);
    expect(limiter.isAllowed('user1')).toBe(true);
    expect(limiter.isAllowed('user1')).toBe(true);
  });

  it('should block requests exceeding limit', () => {
    const limiter = new RateLimiter(2, 1000);
    limiter.isAllowed('user1');
    limiter.isAllowed('user1');
    expect(limiter.isAllowed('user1')).toBe(false);
  });

  it('should reset rate limit for specific key', () => {
    const limiter = new RateLimiter(1, 1000);
    limiter.isAllowed('user1');
    expect(limiter.isAllowed('user1')).toBe(false);
    
    limiter.reset('user1');
    expect(limiter.isAllowed('user1')).toBe(true);
  });
});

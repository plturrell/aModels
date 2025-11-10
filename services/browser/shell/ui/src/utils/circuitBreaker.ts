/**
 * Circuit Breaker pattern implementation
 * Prevents cascading failures by stopping requests when service is down
 */

export interface CircuitBreakerOptions {
  failureThreshold: number; // Number of failures before opening circuit
  resetTimeout: number; // Time in ms before attempting to reset
  monitoringPeriod: number; // Time window for failure counting
}

export enum CircuitState {
  CLOSED = 'closed', // Normal operation
  OPEN = 'open', // Circuit is open, rejecting requests
  HALF_OPEN = 'half-open', // Testing if service recovered
}

export class CircuitBreaker {
  private state: CircuitState = CircuitState.CLOSED;
  private failures: number[] = [];
  private lastFailureTime: number = 0;
  private nextAttemptTime: number = 0;

  constructor(private options: CircuitBreakerOptions) {}

  async execute<T>(fn: () => Promise<T>): Promise<T> {
    if (this.state === CircuitState.OPEN) {
      if (Date.now() < this.nextAttemptTime) {
        throw new Error('Circuit breaker is OPEN. Service unavailable.');
      }
      // Transition to half-open
      this.state = CircuitState.HALF_OPEN;
    }

    try {
      const result = await fn();
      this.onSuccess();
      return result;
    } catch (error) {
      this.onFailure();
      throw error;
    }
  }

  private onSuccess(): void {
    this.failures = [];
    if (this.state === CircuitState.HALF_OPEN) {
      this.state = CircuitState.CLOSED;
    }
  }

  private onFailure(): void {
    const now = Date.now();
    this.lastFailureTime = now;
    
    // Remove failures outside monitoring period
    this.failures = this.failures.filter(
      (time) => now - time < this.options.monitoringPeriod
    );
    
    this.failures.push(now);

    if (this.failures.length >= this.options.failureThreshold) {
      this.state = CircuitState.OPEN;
      this.nextAttemptTime = now + this.options.resetTimeout;
    }
  }

  getState(): CircuitState {
    return this.state;
  }

  reset(): void {
    this.state = CircuitState.CLOSED;
    this.failures = [];
    this.lastFailureTime = 0;
    this.nextAttemptTime = 0;
  }
}

/**
 * Create a circuit breaker with default options
 */
export function createCircuitBreaker(options?: Partial<CircuitBreakerOptions>): CircuitBreaker {
  return new CircuitBreaker({
    failureThreshold: 5,
    resetTimeout: 30000, // 30 seconds
    monitoringPeriod: 60000, // 1 minute
    ...options,
  });
}


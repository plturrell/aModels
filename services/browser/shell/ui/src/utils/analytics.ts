/**
 * Common analytics utility functions
 * Statistical functions, data transformation helpers, and analytics helpers
 */

/**
 * Calculate mean (average) of a number array
 */
export function mean(values: number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((sum, val) => sum + val, 0) / values.length;
}

/**
 * Calculate median of a number array
 */
export function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[mid - 1] + sorted[mid]) / 2
    : sorted[mid];
}

/**
 * Calculate standard deviation
 */
export function standardDeviation(values: number[]): number {
  if (values.length === 0) return 0;
  const avg = mean(values);
  const squareDiffs = values.map((val) => Math.pow(val - avg, 2));
  const avgSquareDiff = mean(squareDiffs);
  return Math.sqrt(avgSquareDiff);
}

/**
 * Calculate percentile
 */
export function percentile(values: number[], p: number): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const index = (p / 100) * (sorted.length - 1);
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  const weight = index - lower;
  return sorted[lower] * (1 - weight) + sorted[upper] * weight;
}

/**
 * Calculate correlation coefficient between two arrays
 */
export function correlation(x: number[], y: number[]): number {
  if (x.length !== y.length || x.length === 0) return 0;
  
  const meanX = mean(x);
  const meanY = mean(y);
  
  let numerator = 0;
  let sumSqX = 0;
  let sumSqY = 0;
  
  for (let i = 0; i < x.length; i++) {
    const diffX = x[i] - meanX;
    const diffY = y[i] - meanY;
    numerator += diffX * diffY;
    sumSqX += diffX * diffX;
    sumSqY += diffY * diffY;
  }
  
  const denominator = Math.sqrt(sumSqX * sumSqY);
  return denominator === 0 ? 0 : numerator / denominator;
}

/**
 * Group data by a key function
 */
export function groupBy<T>(
  data: T[],
  keyFn: (item: T) => string | number
): Record<string, T[]> {
  return data.reduce((acc, item) => {
    const key = String(keyFn(item));
    if (!acc[key]) {
      acc[key] = [];
    }
    acc[key].push(item);
    return acc;
  }, {} as Record<string, T[]>);
}

/**
 * Aggregate grouped data
 */
export function aggregate<T, R>(
  groups: Record<string, T[]>,
  aggregator: (group: T[]) => R
): Record<string, R> {
  return Object.entries(groups).reduce((acc, [key, group]) => {
    acc[key] = aggregator(group);
    return acc;
  }, {} as Record<string, R>);
}

/**
 * Transform time series data to different intervals
 */
export function resampleTimeSeries(
  data: Array<{ timestamp: number | string | Date; value: number }>,
  interval: 'hour' | 'day' | 'week' | 'month'
): Array<{ timestamp: number; value: number }> {
  const intervalMs: Record<string, number> = {
    hour: 60 * 60 * 1000,
    day: 24 * 60 * 60 * 1000,
    week: 7 * 24 * 60 * 60 * 1000,
    month: 30 * 24 * 60 * 60 * 1000,
  };

  const intervalSize = intervalMs[interval];
  if (!intervalSize) return [];

  const grouped = groupBy(data, (item) => {
    const ts = typeof item.timestamp === 'string'
      ? new Date(item.timestamp).getTime()
      : item.timestamp instanceof Date
      ? item.timestamp.getTime()
      : item.timestamp;
    return Math.floor(ts / intervalSize) * intervalSize;
  });

  return Object.entries(grouped).map(([timestamp, group]) => ({
    timestamp: Number(timestamp),
    value: mean(group.map((g) => g.value)),
  }));
}

/**
 * Calculate moving average
 */
export function movingAverage(values: number[], windowSize: number): number[] {
  if (values.length === 0 || windowSize <= 0) return [];
  if (windowSize >= values.length) return [mean(values)];

  const result: number[] = [];
  for (let i = 0; i <= values.length - windowSize; i++) {
    const window = values.slice(i, i + windowSize);
    result.push(mean(window));
  }
  return result;
}

/**
 * Detect anomalies using z-score
 */
export function detectAnomalies(
  values: number[],
  threshold: number = 3
): Array<{ index: number; value: number; zScore: number }> {
  if (values.length === 0) return [];
  
  const avg = mean(values);
  const std = standardDeviation(values);
  
  if (std === 0) return [];
  
  return values
    .map((value, index) => ({
      index,
      value,
      zScore: Math.abs((value - avg) / std),
    }))
    .filter((item) => item.zScore > threshold);
}

/**
 * Format number with appropriate units
 */
export function formatNumber(
  value: number,
  decimals: number = 2,
  unit: string = ''
): string {
  if (value === 0) return `0${unit ? ` ${unit}` : ''}`;
  
  const absValue = Math.abs(value);
  let formatted: string;
  
  if (absValue >= 1e9) {
    formatted = (value / 1e9).toFixed(decimals) + 'B';
  } else if (absValue >= 1e6) {
    formatted = (value / 1e6).toFixed(decimals) + 'M';
  } else if (absValue >= 1e3) {
    formatted = (value / 1e3).toFixed(decimals) + 'K';
  } else {
    formatted = value.toFixed(decimals);
  }
  
  return formatted + (unit ? ` ${unit}` : '');
}

/**
 * Format percentage
 */
export function formatPercentage(value: number, decimals: number = 1): string {
  return `${(value * 100).toFixed(decimals)}%`;
}

/**
 * Calculate growth rate
 */
export function growthRate(current: number, previous: number): number {
  if (previous === 0) return current > 0 ? 1 : 0;
  return (current - previous) / previous;
}

/**
 * Calculate compound annual growth rate (CAGR)
 */
export function cagr(
  startValue: number,
  endValue: number,
  periods: number
): number {
  if (startValue === 0 || periods === 0) return 0;
  return Math.pow(endValue / startValue, 1 / periods) - 1;
}

/**
 * Normalize data to 0-1 range
 */
export function normalize(values: number[]): number[] {
  if (values.length === 0) return [];
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min;
  if (range === 0) return values.map(() => 0.5);
  return values.map((v) => (v - min) / range);
}

/**
 * Calculate trend direction
 */
export function calculateTrend(values: number[]): 'up' | 'down' | 'stable' {
  if (values.length < 2) return 'stable';
  
  const firstHalf = values.slice(0, Math.floor(values.length / 2));
  const secondHalf = values.slice(Math.floor(values.length / 2));
  
  const firstAvg = mean(firstHalf);
  const secondAvg = mean(secondHalf);
  
  const diff = secondAvg - firstAvg;
  const threshold = Math.abs(firstAvg) * 0.05; // 5% threshold
  
  if (Math.abs(diff) < threshold) return 'stable';
  return diff > 0 ? 'up' : 'down';
}

/**
 * Calculate confidence interval
 */
export function confidenceInterval(
  values: number[],
  confidence: number = 0.95
): { lower: number; upper: number; mean: number } {
  if (values.length === 0) {
    return { lower: 0, upper: 0, mean: 0 };
  }
  
  const avg = mean(values);
  const std = standardDeviation(values);
  const n = values.length;
  
  // Z-score for confidence level (approximation)
  const zScores: Record<number, number> = {
    0.9: 1.645,
    0.95: 1.96,
    0.99: 2.576,
  };
  const z = zScores[confidence] || 1.96;
  
  const margin = (z * std) / Math.sqrt(n);
  
  return {
    mean: avg,
    lower: avg - margin,
    upper: avg + margin,
  };
}

/**
 * Calculate rate of change
 */
export function rateOfChange(values: number[]): number[] {
  if (values.length < 2) return [];
  
  const changes: number[] = [];
  for (let i = 1; i < values.length; i++) {
    const change = values[i] - values[i - 1];
    changes.push(change);
  }
  return changes;
}

/**
 * Calculate percentage change
 */
export function percentageChange(values: number[]): number[] {
  if (values.length < 2) return [];
  
  const changes: number[] = [];
  for (let i = 1; i < values.length; i++) {
    if (values[i - 1] === 0) {
      changes.push(values[i] > 0 ? 1 : 0);
    } else {
      changes.push((values[i] - values[i - 1]) / values[i - 1]);
    }
  }
  return changes;
}


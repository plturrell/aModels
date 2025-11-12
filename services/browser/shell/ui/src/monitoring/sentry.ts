/**
 * Sentry Error Tracking Configuration
 * 
 * Monitors application errors, performance, and user sessions
 */

import * as Sentry from "@sentry/react";

export function initSentry() {
  // Only initialize in production or when explicitly enabled
  if (import.meta.env.MODE !== 'production' && !import.meta.env.VITE_ENABLE_SENTRY) {
    console.log('Sentry disabled in development mode');
    return;
  }

  const dsn = import.meta.env.VITE_SENTRY_DSN;
  
  if (!dsn) {
    console.warn('Sentry DSN not configured. Set VITE_SENTRY_DSN in .env');
    return;
  }

  Sentry.init({
    dsn,
    environment: import.meta.env.MODE,
    
    // Performance Monitoring
    integrations: [
      Sentry.browserTracingIntegration(),
      
      // Session Replay - captures user sessions for debugging
      Sentry.replayIntegration({
        maskAllText: false, // Set true in production for privacy
        blockAllMedia: false,
        maskAllInputs: true, // Always mask form inputs
      }),
    ],

    // Trace propagation
    tracePropagationTargets: ["localhost", /^\//],

    // Performance Monitoring
    tracesSampleRate: import.meta.env.PROD ? 0.1 : 1.0, // 10% in prod, 100% in dev
    
    // Session Replay
    replaysSessionSampleRate: 0.1, // 10% of sessions
    replaysOnErrorSampleRate: 1.0, // 100% of sessions with errors

    // Custom error filtering
    beforeSend(event, hint) {
      // Don't send errors from browser extensions
      if (event.exception?.values?.[0]?.value?.includes('Extension context')) {
        return null;
      }
      
      // Add custom context
      event.tags = {
        ...event.tags,
        app_version: import.meta.env.VITE_APP_VERSION || 'unknown',
      };
      
      return event;
    },

    // Custom breadcrumb filtering
    beforeBreadcrumb(breadcrumb) {
      // Don't log click events on unimportant elements
      if (breadcrumb.category === 'ui.click' && breadcrumb.message?.includes('div')) {
        return null;
      }
      return breadcrumb;
    },
  });

  console.log('✅ Sentry initialized:', import.meta.env.MODE);
}

/**
 * Manually capture an error with additional context
 */
export function captureError(error: Error, context?: Record<string, any>) {
  Sentry.captureException(error, {
    extra: context,
  });
}

/**
 * Add custom breadcrumb for debugging
 */
export function addBreadcrumb(message: string, data?: Record<string, any>) {
  Sentry.addBreadcrumb({
    message,
    level: 'info',
    data,
  });
}

/**
 * Set user context for tracking
 */
export function setUser(userId: string, email?: string, username?: string) {
  Sentry.setUser({
    id: userId,
    email,
    username,
  });
}

/**
 * Clear user context (on logout)
 */
export function clearUser() {
  Sentry.setUser(null);
}

/**
 * Add custom tags to all future events
 */
export function setTags(tags: Record<string, string>) {
  Sentry.setTags(tags);
}

/**
 * Measure performance of a function
 */
export async function measurePerformance<T>(
  name: string,
  fn: () => Promise<T>
): Promise<T> {
  return await Sentry.startSpan({ name }, async () => {
    return await fn();
  });
}

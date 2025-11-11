/**
 * Error Message Library
 * Maps technical errors to user-friendly messages with recovery actions
 */

const ERROR_MESSAGES = {
  // Connection errors
  'Failed to fetch': {
    title: 'Cannot Connect to Gateway',
    message: 'Unable to reach the gateway server. It may be offline or the URL might be incorrect.',
    actions: [
      'Check if the gateway is running',
      'Verify the gateway URL in Settings',
      'Try restarting the gateway service'
    ],
    icon: '🔌'
  },
  'NetworkError': {
    title: 'Network Error',
    message: 'A network problem prevented the request from completing.',
    actions: [
      'Check your internet connection',
      'Verify firewall settings',
      'Try again in a moment'
    ],
    icon: '🌐'
  },
  
  // HTTP status codes
  400: {
    title: 'Invalid Request',
    message: 'The request format was incorrect. This is likely a bug.',
    actions: [
      'Try the operation again',
      'Check the browser console for details',
      'Report this issue if it persists'
    ],
    icon: '⚠️'
  },
  401: {
    title: 'Authentication Required',
    message: 'You need to authenticate to access this resource.',
    actions: [
      'Check your credentials',
      'Update authentication settings',
      'Contact your administrator'
    ],
    icon: '🔐'
  },
  403: {
    title: 'Access Denied',
    message: 'You don\'t have permission to perform this action.',
    actions: [
      'Verify your permissions',
      'Contact your administrator',
      'Try a different account'
    ],
    icon: '🚫'
  },
  404: {
    title: 'Service Not Found',
    message: 'The requested service endpoint doesn\'t exist.',
    actions: [
      'Verify the gateway is fully started',
      'Check if all services are running',
      'Update to the latest gateway version'
    ],
    icon: '🔍'
  },
  500: {
    title: 'Server Error',
    message: 'The gateway encountered an internal error while processing your request.',
    actions: [
      'Check the gateway logs for details',
      'Try the operation again',
      'Report this error if it persists'
    ],
    icon: '⚙️'
  },
  503: {
    title: 'Service Unavailable',
    message: 'The requested service is temporarily unavailable.',
    actions: [
      'Wait a moment and try again',
      'Check if the service is starting up',
      'Verify all dependencies are running'
    ],
    icon: '⏳'
  },
  
  // Timeout errors
  'timeout': {
    title: 'Request Timed Out',
    message: 'The operation took too long to complete.',
    actions: [
      'Try again with a simpler request',
      'Check if the gateway is overloaded',
      'Increase timeout settings if available'
    ],
    icon: '⏰'
  },
  
  // Default fallback
  'default': {
    title: 'Something Went Wrong',
    message: 'An unexpected error occurred.',
    actions: [
      'Try the operation again',
      'Check the browser console for details',
      'Contact support if the problem persists'
    ],
    icon: '❌'
  }
};

/**
 * Parse error and return user-friendly message
 * @param {Error} error - The error object
 * @returns {Object} User-friendly error details
 */
function getUserFriendlyError(error) {
  // Try to extract status code from error message
  const statusMatch = error.message.match(/status: (\d+)/);
  if (statusMatch) {
    const status = parseInt(statusMatch[1]);
    if (ERROR_MESSAGES[status]) {
      return ERROR_MESSAGES[status];
    }
  }
  
  // Check for known error messages
  for (const [key, value] of Object.entries(ERROR_MESSAGES)) {
    if (error.message.includes(key)) {
      return value;
    }
  }
  
  // Return default error
  return ERROR_MESSAGES.default;
}

/**
 * Format error for display
 * @param {Object} errorInfo - Error information from getUserFriendlyError
 * @returns {string} HTML formatted error message
 */
function formatErrorMessage(errorInfo) {
  const actionsList = errorInfo.actions
    .map(action => `  • ${action}`)
    .join('\n');
  
  return `${errorInfo.icon} ${errorInfo.title}\n${errorInfo.message}\n\nWhat to try:\n${actionsList}`;
}

/**
 * Format success message
 * @param {string} action - The action that succeeded
 * @param {*} data - Optional data to display
 * @returns {string} Formatted success message
 */
function formatSuccessMessage(action, data = null) {
  let message = `✅ ${action} successful`;
  
  if (data && typeof data === 'object') {
    // Format object data nicely
    const keys = Object.keys(data);
    if (keys.length > 0 && keys.length <= 3) {
      const details = keys.map(key => `${key}: ${JSON.stringify(data[key])}`).join(', ');
      message += `\n${details}`;
    } else if (keys.length > 3) {
      message += `\nReturned ${keys.length} fields`;
    }
  } else if (data) {
    message += `\n${data}`;
  }
  
  return message;
}

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { getUserFriendlyError, formatErrorMessage, formatSuccessMessage, ERROR_MESSAGES };
}

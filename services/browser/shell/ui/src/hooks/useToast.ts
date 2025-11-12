/**
 * Toast Notification Hook
 * 
 * Provides a clean API for showing toast notifications
 * using notistack under the hood
 */

import { useSnackbar, VariantType, OptionsObject } from 'notistack';
import { useCallback } from 'react';

interface ToastOptions extends Partial<OptionsObject> {
  /**
   * Custom action button
   */
  action?: React.ReactNode;
  
  /**
   * Auto-hide duration in milliseconds
   */
  duration?: number;
}

export function useToast() {
  const { enqueueSnackbar, closeSnackbar } = useSnackbar();

  const show = useCallback((
    message: string,
    variant: VariantType,
    options?: ToastOptions
  ) => {
    const { duration = 3000, action, ...rest } = options || {};
    
    return enqueueSnackbar(message, {
      variant,
      autoHideDuration: duration,
      action,
      ...rest,
    });
  }, [enqueueSnackbar]);

  return {
    /**
     * Show success message
     */
    success: useCallback((message: string, options?: ToastOptions) => {
      return show(message, 'success', options);
    }, [show]),

    /**
     * Show error message
     */
    error: useCallback((message: string, options?: ToastOptions) => {
      return show(message, 'error', options);
    }, [show]),

    /**
     * Show info message
     */
    info: useCallback((message: string, options?: ToastOptions) => {
      return show(message, 'info', options);
    }, [show]),

    /**
     * Show warning message
     */
    warning: useCallback((message: string, options?: ToastOptions) => {
      return show(message, 'warning', options);
    }, [show]),

    /**
     * Show loading message that must be manually closed
     */
    loading: useCallback((message: string) => {
      return show(message, 'info', {
        duration: null, // Don't auto-hide
        persist: true,
      });
    }, [show]),

    /**
     * Update an existing toast
     */
    update: useCallback((key: string | number, message: string, variant: VariantType) => {
      closeSnackbar(key);
      return show(message, variant);
    }, [show, closeSnackbar]),

    /**
     * Close a specific toast
     */
    close: closeSnackbar,

    /**
     * Show a promise-based toast with loading/success/error states
     */
    promise: useCallback(async <T,>(
      promise: Promise<T>,
      {
        loading: loadingMsg = 'Processing...',
        success: successMsg = 'Success!',
        error: errorMsg = 'Something went wrong',
      }: {
        loading?: string;
        success?: string | ((data: T) => string);
        error?: string | ((error: any) => string);
      }
    ): Promise<T> => {
      const key = show(loadingMsg, 'info', { persist: true });
      
      try {
        const result = await promise;
        closeSnackbar(key);
        const msg = typeof successMsg === 'function' ? successMsg(result) : successMsg;
        show(msg, 'success');
        return result;
      } catch (err) {
        closeSnackbar(key);
        const msg = typeof errorMsg === 'function' ? errorMsg(err) : errorMsg;
        show(msg, 'error', { duration: 5000 });
        throw err;
      }
    }, [show, closeSnackbar]),
  };
}

/**
 * Example usage:
 * 
 * const toast = useToast();
 * 
 * // Simple
 * toast.success('Graph loaded!');
 * toast.error('Failed to load data');
 * 
 * // With options
 * toast.info('Processing...', { duration: 5000 });
 * 
 * // With action button
 * toast.error('Failed to save', {
 *   action: <Button onClick={retry}>Retry</Button>
 * });
 * 
 * // Promise-based
 * await toast.promise(
 *   loadGraphData(),
 *   {
 *     loading: 'Loading graph...',
 *     success: 'Graph loaded successfully!',
 *     error: 'Failed to load graph',
 *   }
 * );
 * 
 * // Update existing toast
 * const key = toast.loading('Uploading...');
 * // ... later
 * toast.update(key, 'Upload complete!', 'success');
 */

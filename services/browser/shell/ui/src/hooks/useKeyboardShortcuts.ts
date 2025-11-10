/**
 * Keyboard shortcuts hook for improved UX
 * Provides global keyboard shortcuts for common actions
 */

import { useEffect, useCallback } from 'react';

export interface KeyboardShortcut {
  key: string;
  ctrlKey?: boolean;
  metaKey?: boolean;
  shiftKey?: boolean;
  altKey?: boolean;
  action: () => void;
  description: string;
}

export function useKeyboardShortcuts(shortcuts: KeyboardShortcut[]) {
  const handleKeyDown = useCallback(
    (event: KeyboardEvent) => {
      for (const shortcut of shortcuts) {
        const keyMatch = event.key.toLowerCase() === shortcut.key.toLowerCase();
        const ctrlMatch = shortcut.ctrlKey === undefined || event.ctrlKey === shortcut.ctrlKey;
        const metaMatch = shortcut.metaKey === undefined || event.metaKey === shortcut.metaKey;
        const shiftMatch = shortcut.shiftKey === undefined || event.shiftKey === shortcut.shiftKey;
        const altMatch = shortcut.altKey === undefined || event.altKey === shortcut.altKey;

        if (keyMatch && ctrlMatch && metaMatch && shiftMatch && altMatch) {
          // Don't trigger if typing in input/textarea
          const target = event.target as HTMLElement;
          if (
            target.tagName === 'INPUT' ||
            target.tagName === 'TEXTAREA' ||
            target.isContentEditable
          ) {
            continue;
          }

          event.preventDefault();
          shortcut.action();
          break;
        }
      }
    },
    [shortcuts]
  );

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [handleKeyDown]);
}

/**
 * Common keyboard shortcuts for analytics dashboards
 */
export const DASHBOARD_SHORTCUTS: KeyboardShortcut[] = [
  {
    key: 'f',
    ctrlKey: false,
    metaKey: false,
    description: 'Focus search/filter',
    action: () => {
      const searchInput = document.querySelector<HTMLInputElement>('input[type="search"], input[placeholder*="search" i], input[placeholder*="filter" i]');
      searchInput?.focus();
    },
  },
  {
    key: 'r',
    ctrlKey: false,
    metaKey: false,
    description: 'Refresh dashboard',
    action: () => {
      window.dispatchEvent(new CustomEvent('dashboard-refresh'));
    },
  },
  {
    key: 'e',
    ctrlKey: true,
    description: 'Export dashboard',
    action: () => {
      window.dispatchEvent(new CustomEvent('dashboard-export'));
    },
  },
  {
    key: '?',
    ctrlKey: false,
    metaKey: false,
    shiftKey: true,
    description: 'Show keyboard shortcuts',
    action: () => {
      window.dispatchEvent(new CustomEvent('show-shortcuts'));
    },
  },
];


/**
 * Global Keyboard Shortcuts Hook
 * 
 * Provides application-wide keyboard shortcuts for navigation and actions
 */

import { useEffect } from 'react';

export const GLOBAL_SHORTCUTS = {
  'Mod+1': 'Navigate to Home',
  'Mod+2': 'Navigate to Graph',
  'Mod+3': 'Navigate to Extract',
  'Mod+4': 'Navigate to Training',
  'Mod+5': 'Navigate to Postgres',
  'Mod+6': 'Navigate to LocalAI',
  'Mod+7': 'Navigate to DMS',
  'Mod+8': 'Navigate to SAP',
  'Mod+K': 'Open Command Palette',
  'Escape': 'Close Modal/Dialog',
  '?': 'Show Keyboard Shortcuts',
} as const;

export type ShortcutAction = 
  | { type: 'navigate'; moduleId: string }
  | { type: 'command-palette' }
  | { type: 'escape' }
  | { type: 'help' };

interface UseGlobalShortcutsOptions {
  onNavigate: (moduleId: string) => void;
  onCommandPalette?: () => void;
  onEscape?: () => void;
  onHelp?: () => void;
  enabled?: boolean;
}

export function useGlobalShortcuts({
  onNavigate,
  onCommandPalette,
  onEscape,
  onHelp,
  enabled = true,
}: UseGlobalShortcutsOptions) {
  useEffect(() => {
    if (!enabled) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      const isMod = e.ctrlKey || e.metaKey;
      const isShift = e.shiftKey;

      // Ignore if user is typing in an input
      const target = e.target as HTMLElement;
      if (
        target.tagName === 'INPUT' ||
        target.tagName === 'TEXTAREA' ||
        target.isContentEditable
      ) {
        // Allow Escape and Cmd+K even in inputs
        if (e.key !== 'Escape' && !(isMod && e.key === 'k')) {
          return;
        }
      }

      // Module navigation (Cmd+1 through Cmd+8)
      if (isMod && e.key >= '1' && e.key <= '8') {
        e.preventDefault();
        const modules = [
          'home',    // Cmd+1
          'graph',   // Cmd+2
          'extract', // Cmd+3
          'training',// Cmd+4
          'postgres',// Cmd+5
          'localai', // Cmd+6
          'dms',     // Cmd+7
          'sap',     // Cmd+8
        ];
        const moduleIndex = parseInt(e.key) - 1;
        if (modules[moduleIndex]) {
          onNavigate(modules[moduleIndex]);
        }
        return;
      }

      // Command palette (Cmd+K)
      if (isMod && e.key === 'k') {
        e.preventDefault();
        onCommandPalette?.();
        return;
      }

      // Escape key
      if (e.key === 'Escape') {
        e.preventDefault();
        onEscape?.();
        return;
      }

      // Help dialog (?)
      if (e.key === '?' && !isMod && !isShift) {
        e.preventDefault();
        onHelp?.();
        return;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onNavigate, onCommandPalette, onEscape, onHelp, enabled]);
}

/**
 * Format shortcut key for display
 */
export function formatShortcut(shortcut: string): string {
  const isMac = navigator.platform.toUpperCase().indexOf('MAC') >= 0;
  return shortcut
    .replace('Mod', isMac ? '⌘' : 'Ctrl')
    .replace('+', isMac ? '' : '+');
}

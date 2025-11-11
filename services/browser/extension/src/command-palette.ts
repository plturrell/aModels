/**
 * Command Palette - Keyboard-driven action launcher
 * TypeScript implementation following Apple standards
 * Keyboard: Cmd/Ctrl+K to open
 */

import type { Command } from './types';

export class CommandPalette {
  private isOpen = false;
  private commands: Command[] = [];
  private filteredCommands: Command[] = [];
  private selectedIndex = 0;
  private recentActions: string[] = [];
  
  private backdrop!: HTMLDivElement;
  private container!: HTMLDivElement;
  private searchInput!: HTMLInputElement;
  private resultsContainer!: HTMLDivElement;
  private footer!: HTMLDivElement;

  constructor() {
    this.recentActions = this.loadRecentActions();
    this.createElements();
    this.registerCommands();
    this.attachEventListeners();
  }

  private createElements(): void {
    // Backdrop
    this.backdrop = document.createElement('div');
    this.backdrop.id = 'command-palette-backdrop';
    this.backdrop.className = 'command-palette-backdrop';
    this.backdrop.setAttribute('role', 'presentation');
    
    // Container
    this.container = document.createElement('div');
    this.container.id = 'command-palette';
    this.container.className = 'command-palette';
    this.container.setAttribute('role', 'dialog');
    this.container.setAttribute('aria-modal', 'true');
    this.container.setAttribute('aria-labelledby', 'command-palette-title');
    
    // Search input
    this.searchInput = document.createElement('input');
    this.searchInput.type = 'text';
    this.searchInput.className = 'command-palette-input';
    this.searchInput.placeholder = 'Type a command or search...';
    this.searchInput.setAttribute('aria-label', 'Command search');
    this.searchInput.setAttribute('autocomplete', 'off');
    this.searchInput.setAttribute('spellcheck', 'false');
    
    // Results container
    this.resultsContainer = document.createElement('div');
    this.resultsContainer.className = 'command-palette-results';
    this.resultsContainer.setAttribute('role', 'listbox');
    this.resultsContainer.setAttribute('aria-label', 'Available commands');
    
    // Footer hint
    this.footer = document.createElement('div');
    this.footer.className = 'command-palette-footer';
    this.footer.innerHTML = `
      <span><kbd>↑↓</kbd> Navigate</span>
      <span><kbd>Enter</kbd> Execute</span>
      <span><kbd>Esc</kbd> Close</span>
    `;
    
    // Assemble
    this.container.appendChild(this.searchInput);
    this.container.appendChild(this.resultsContainer);
    this.container.appendChild(this.footer);
    
    // Append to body (hidden initially)
    document.body.appendChild(this.backdrop);
    document.body.appendChild(this.container);
  }

  private registerCommands(): void {
    this.commands = [
      // Quick actions
      {
        id: 'extract-text',
        name: 'Extract Text from Page',
        description: 'Run OCR extraction on the current page',
        icon: '📄',
        keywords: ['ocr', 'extract', 'text', 'read'],
        action: () => window.runOcr?.(),
        category: 'Quick Actions'
      },
      {
        id: 'query-data',
        name: 'Query Data',
        description: 'Run SQL query on your database',
        icon: '🔍',
        keywords: ['sql', 'query', 'data', 'database'],
        action: () => window.runSql?.(),
        category: 'Quick Actions'
      },
      {
        id: 'view-telemetry',
        name: 'View Telemetry',
        description: 'Get recent telemetry and metrics',
        icon: '📊',
        keywords: ['telemetry', 'metrics', 'stats', 'monitor'],
        action: () => window.runTelemetry?.(),
        category: 'Quick Actions'
      },
      {
        id: 'open-browser',
        name: 'Open Browser Shell',
        description: 'Launch the Electron browser interface',
        icon: '🌐',
        keywords: ['browser', 'shell', 'electron', 'open'],
        action: () => window.openBrowser?.(),
        category: 'Quick Actions'
      },
      {
        id: 'run-agentflow',
        name: 'Run AgentFlow',
        description: 'Execute an agent workflow',
        icon: '⚡',
        keywords: ['agent', 'flow', 'workflow', 'automation'],
        action: () => window.runAgentFlow?.(),
        category: 'Advanced'
      },
      {
        id: 'opensearch',
        name: 'OpenSearch Query',
        description: 'Search using OpenSearch',
        icon: '🔎',
        keywords: ['search', 'opensearch', 'elasticsearch', 'query'],
        action: () => window.runSearch?.(),
        category: 'Advanced'
      },
      {
        id: 'open-settings',
        name: 'Open Settings',
        description: 'Configure gateway and browser URLs',
        icon: '⚙️',
        keywords: ['settings', 'options', 'config', 'preferences'],
        action: () => (window as any).chrome?.runtime?.openOptionsPage(),
        category: 'Navigation'
      }
    ];
  }

  private attachEventListeners(): void {
    // Keyboard shortcut (Cmd/Ctrl+K)
    document.addEventListener('keydown', (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        this.isOpen ? this.close() : this.open();
      }
      
      if (e.key === 'Escape' && this.isOpen) {
        this.close();
      }
    });

    // Search input
    this.searchInput.addEventListener('input', (e) => {
      const target = e.target as HTMLInputElement;
      this.handleSearch(target.value);
    });

    this.searchInput.addEventListener('keydown', (e: KeyboardEvent) => {
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        this.selectedIndex = Math.min(
          this.selectedIndex + 1,
          this.filteredCommands.length - 1
        );
        this.renderResults();
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        this.selectedIndex = Math.max(this.selectedIndex - 1, 0);
        this.renderResults();
      } else if (e.key === 'Enter') {
        e.preventDefault();
        const command = this.filteredCommands[this.selectedIndex];
        if (command) {
          this.executeCommand(command);
        }
      }
    });

    // Backdrop click to close
    this.backdrop.addEventListener('click', () => this.close());
  }

  private handleSearch(query: string): void {
    const lowerQuery = query.toLowerCase().trim();
    
    if (!lowerQuery) {
      // Show recent actions when no query
      const recentCommands = this.commands.filter((cmd) =>
        this.recentActions.includes(cmd.id)
      );
      this.filteredCommands = recentCommands.length ? recentCommands : this.commands;
    } else {
      // Filter by name, description, and keywords
      this.filteredCommands = this.commands.filter((cmd) => {
        return (
          cmd.name.toLowerCase().includes(lowerQuery) ||
          cmd.description.toLowerCase().includes(lowerQuery) ||
          cmd.keywords.some((kw) => kw.toLowerCase().includes(lowerQuery))
        );
      });
    }

    this.selectedIndex = 0;
    this.renderResults();
  }

  private renderResults(): void {
    this.resultsContainer.innerHTML = '';

    if (this.filteredCommands.length === 0) {
      const noResults = document.createElement('div');
      noResults.className = 'command-palette-no-results';
      noResults.textContent = 'No commands found';
      noResults.setAttribute('role', 'status');
      this.resultsContainer.appendChild(noResults);
      return;
    }

    this.filteredCommands.forEach((cmd, index) => {
      const item = document.createElement('div');
      item.className = 'command-palette-item';
      item.setAttribute('role', 'option');
      item.setAttribute('aria-selected', (index === this.selectedIndex).toString());
      
      if (index === this.selectedIndex) {
        item.classList.add('selected');
        item.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
      }

      item.innerHTML = `
        <span class="command-palette-icon">${cmd.icon}</span>
        <div class="command-palette-text">
          <div class="command-palette-name">${cmd.name}</div>
          <div class="command-palette-description">${cmd.description}</div>
        </div>
        <span class="command-palette-category">${cmd.category}</span>
      `;

      item.addEventListener('click', () => this.executeCommand(cmd));
      item.addEventListener('mouseenter', () => {
        this.selectedIndex = index;
        this.renderResults();
      });

      this.resultsContainer.appendChild(item);
    });
  }

  private async executeCommand(command: Command): Promise<void> {
    this.close();
    this.saveRecentAction(command.id);

    try {
      await command.action();
    } catch (error) {
      console.error(`Command "${command.id}" failed:`, error);
    }
  }

  private loadRecentActions(): string[] {
    const stored = localStorage.getItem('commandPaletteRecent');
    return stored ? JSON.parse(stored) : [];
  }

  private saveRecentAction(commandId: string): void {
    this.recentActions = this.recentActions.filter((id) => id !== commandId);
    this.recentActions.unshift(commandId);
    this.recentActions = this.recentActions.slice(0, 5);
    localStorage.setItem('commandPaletteRecent', JSON.stringify(this.recentActions));
  }

  public open(): void {
    this.isOpen = true;
    this.backdrop.classList.add('visible');
    this.container.classList.add('visible');
    this.searchInput.value = '';
    this.handleSearch('');
    setTimeout(() => this.searchInput.focus(), 100);
  }

  public close(): void {
    this.isOpen = false;
    this.backdrop.classList.remove('visible');
    this.container.classList.remove('visible');
    this.searchInput.value = '';
  }
}

// Initialize on DOM ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    window.commandPalette = new CommandPalette();
  });
} else {
  window.commandPalette = new CommandPalette();
}

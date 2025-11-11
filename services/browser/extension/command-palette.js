/**
 * Command Palette - Keyboard-driven action launcher
 * Inspired by macOS Spotlight and VS Code Command Palette
 * Keyboard: Cmd/Ctrl+K to open
 */

class CommandPalette {
  constructor() {
    this.isOpen = false;
    this.commands = [];
    this.filteredCommands = [];
    this.selectedIndex = 0;
    this.recentActions = this.loadRecentActions();
    
    this.createElements();
    this.registerCommands();
    this.attachEventListeners();
  }
  
  createElements() {
    // Backdrop
    this.backdrop = document.createElement('div');
    this.backdrop.id = 'command-palette-backdrop';
    this.backdrop.className = 'command-palette-backdrop';
    
    // Container
    this.container = document.createElement('div');
    this.container.id = 'command-palette';
    this.container.className = 'command-palette';
    
    // Search input
    this.searchInput = document.createElement('input');
    this.searchInput.type = 'text';
    this.searchInput.className = 'command-palette-input';
    this.searchInput.placeholder = 'Type a command or search...';
    this.searchInput.setAttribute('aria-label', 'Command search');
    
    // Results container
    this.resultsContainer = document.createElement('div');
    this.resultsContainer.className = 'command-palette-results';
    this.resultsContainer.setAttribute('role', 'listbox');
    
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
    
    // Append to body (will be hidden initially)
    document.body.appendChild(this.backdrop);
    document.body.appendChild(this.container);
  }
  
  registerCommands() {
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
      
      // Advanced tools
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
        id: 'redis-set',
        name: 'Redis Set Value',
        description: 'Store a value in Redis cache',
        icon: '💾',
        keywords: ['redis', 'cache', 'set', 'store'],
        action: () => window.redisSet?.(),
        category: 'Advanced'
      },
      {
        id: 'redis-get',
        name: 'Redis Get Value',
        description: 'Retrieve a value from Redis cache',
        icon: '📥',
        keywords: ['redis', 'cache', 'get', 'retrieve'],
        action: () => window.redisGet?.(),
        category: 'Advanced'
      },
      
      // Navigation
      {
        id: 'open-settings',
        name: 'Open Settings',
        description: 'Configure gateway and browser URLs',
        icon: '⚙️',
        keywords: ['settings', 'options', 'config', 'preferences'],
        action: () => chrome.runtime.openOptionsPage(),
        category: 'Navigation'
      },
      {
        id: 'view-help',
        name: 'View Documentation',
        description: 'Open help and documentation',
        icon: '❓',
        keywords: ['help', 'docs', 'documentation', 'guide'],
        action: () => chrome.tabs.create({ url: 'https://github.com/yourusername/aModels' }),
        category: 'Navigation'
      },
      {
        id: 'check-connection',
        name: 'Check Gateway Connection',
        description: 'Test connection to gateway',
        icon: '🔌',
        keywords: ['check', 'test', 'connection', 'health', 'gateway'],
        action: () => window.checkConnection?.(),
        category: 'System'
      },
      {
        id: 'toggle-advanced',
        name: 'Toggle Advanced Tools',
        description: 'Show or hide advanced features',
        icon: '🔧',
        keywords: ['advanced', 'toggle', 'show', 'hide'],
        action: () => window.toggleAdvanced?.(),
        category: 'System'
      },
      
      // Analytics commands
      {
        id: 'view-analytics',
        name: 'View Analytics Dashboard',
        description: 'Open analytics dashboard in browser shell',
        icon: '📊',
        keywords: ['analytics', 'dashboard', 'metrics', 'stats', 'view'],
        action: async () => {
          const { browserUrl } = await chrome.storage.sync.get(['browserUrl']);
          const url = browserUrl || 'http://localhost:8070';
          chrome.tabs.create({ url: `${url}/#/analytics` });
        },
        category: 'Analytics'
      },
      {
        id: 'ask-analytics',
        name: 'Ask About Analytics',
        description: 'Ask AI questions about your analytics data',
        icon: '🤖',
        keywords: ['ask', 'analytics', 'ai', 'question', 'insight'],
        action: async () => {
          const { browserUrl } = await chrome.storage.sync.get(['browserUrl']);
          const url = browserUrl || 'http://localhost:8070';
          chrome.tabs.create({ url: `${url}/#/analytics` });
          // Focus on AI assistant after a delay
          setTimeout(() => {
            window.postMessage({ type: 'focus-analytics-ai' }, '*');
          }, 1000);
        },
        category: 'Analytics'
      },
      
      // Search commands
      {
        id: 'open-search',
        name: 'Open Search',
        description: 'Open search interface in browser shell',
        icon: '🔍',
        keywords: ['search', 'find', 'query', 'lookup'],
        action: async () => {
          const { browserUrl } = await chrome.storage.sync.get(['browserUrl']);
          const url = browserUrl || 'http://localhost:8070';
          chrome.tabs.create({ url: `${url}/#/search` });
        },
        category: 'Search'
      },
      {
        id: 'quick-search',
        name: 'Quick Search',
        description: 'Perform a quick search from extension',
        icon: '⚡',
        keywords: ['quick', 'search', 'fast', 'find'],
        action: () => {
          const input = document.getElementById('quick-search-input');
          if (input) {
            input.focus();
            this.close();
          }
        },
        category: 'Search'
      },
      {
        id: 'ai-search',
        name: 'AI-Enhanced Search',
        description: 'Use AI to enhance your search query',
        icon: '🧠',
        keywords: ['ai', 'search', 'enhance', 'improve', 'refine'],
        action: async () => {
          const { browserUrl } = await chrome.storage.sync.get(['browserUrl']);
          const url = browserUrl || 'http://localhost:8070';
          chrome.tabs.create({ url: `${url}/#/search` });
          setTimeout(() => {
            window.postMessage({ type: 'focus-search-ai' }, '*');
          }, 1000);
        },
        category: 'Search'
      },
      
      // LocalAI commands
      {
        id: 'open-chat',
        name: 'Open LocalAI Chat',
        description: 'Open LocalAI chat interface',
        icon: '💬',
        keywords: ['chat', 'localai', 'ai', 'conversation', 'talk'],
        action: async () => {
          const { browserUrl } = await chrome.storage.sync.get(['browserUrl']);
          const url = browserUrl || 'http://localhost:8070';
          chrome.tabs.create({ url: `${url}/#/localai` });
        },
        category: 'LocalAI'
      },
      {
        id: 'quick-chat',
        name: 'Quick Chat',
        description: 'Start a quick chat from extension',
        icon: '⚡',
        keywords: ['quick', 'chat', 'fast', 'message'],
        action: () => {
          const input = document.getElementById('chat-input');
          if (input) {
            input.focus();
            this.close();
          }
        },
        category: 'LocalAI'
      },
      {
        id: 'ask-about-search',
        name: 'Ask AI About Search Results',
        description: 'Ask LocalAI questions about your search results',
        icon: '🤔',
        keywords: ['ask', 'search', 'results', 'ai', 'question'],
        action: async () => {
          const { browserUrl } = await chrome.storage.sync.get(['browserUrl']);
          const url = browserUrl || 'http://localhost:8070';
          chrome.tabs.create({ url: `${url}/#/search` });
          setTimeout(() => {
            window.postMessage({ type: 'focus-search-ai-ask' }, '*');
          }, 1000);
        },
        category: 'LocalAI'
      },
      {
        id: 'ask-about-analytics',
        name: 'Ask AI About Analytics',
        description: 'Ask LocalAI questions about your analytics',
        icon: '📈',
        keywords: ['ask', 'analytics', 'ai', 'question', 'insight'],
        action: async () => {
          const { browserUrl } = await chrome.storage.sync.get(['browserUrl']);
          const url = browserUrl || 'http://localhost:8070';
          chrome.tabs.create({ url: `${url}/#/analytics` });
          setTimeout(() => {
            window.postMessage({ type: 'focus-analytics-ai' }, '*');
          }, 1000);
        },
        category: 'LocalAI'
      }
    ];
  }
  
  attachEventListeners() {
    // Global keyboard shortcut: Cmd/Ctrl+K
    document.addEventListener('keydown', (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        this.toggle();
      }
    });
    
    // Backdrop click to close
    this.backdrop.addEventListener('click', () => this.close());
    
    // Search input
    this.searchInput.addEventListener('input', (e) => {
      this.handleSearch(e.target.value);
    });
    
    // Keyboard navigation
    this.searchInput.addEventListener('keydown', (e) => {
      switch (e.key) {
        case 'ArrowDown':
          e.preventDefault();
          this.selectNext();
          break;
        case 'ArrowUp':
          e.preventDefault();
          this.selectPrevious();
          break;
        case 'Enter':
          e.preventDefault();
          this.executeSelected();
          break;
        case 'Escape':
          e.preventDefault();
          this.close();
          break;
      }
    });
  }
  
  handleSearch(query) {
    query = query.toLowerCase().trim();
    
    if (!query) {
      // Show recent actions + all commands
      this.filteredCommands = [
        ...this.getRecentCommands(),
        ...this.commands
      ];
    } else {
      // Fuzzy search
      this.filteredCommands = this.commands.filter(cmd => {
        const searchText = `${cmd.name} ${cmd.description} ${cmd.keywords.join(' ')}`.toLowerCase();
        return searchText.includes(query);
      });
      
      // Sort by relevance (name matches first)
      this.filteredCommands.sort((a, b) => {
        const aNameMatch = a.name.toLowerCase().includes(query);
        const bNameMatch = b.name.toLowerCase().includes(query);
        if (aNameMatch && !bNameMatch) return -1;
        if (!aNameMatch && bNameMatch) return 1;
        return 0;
      });
    }
    
    this.selectedIndex = 0;
    this.render();
  }
  
  getRecentCommands() {
    return this.recentActions
      .map(id => this.commands.find(cmd => cmd.id === id))
      .filter(Boolean)
      .map(cmd => ({ ...cmd, isRecent: true }));
  }
  
  render() {
    this.resultsContainer.innerHTML = '';
    
    if (this.filteredCommands.length === 0) {
      this.resultsContainer.innerHTML = `
        <div class="command-palette-empty">
          <span>No commands found</span>
        </div>
      `;
      return;
    }
    
    // Group by category
    const grouped = this.groupByCategory(this.filteredCommands);
    
    let index = 0;
    for (const [category, commands] of Object.entries(grouped)) {
      // Category header
      if (category !== 'null') {
        const header = document.createElement('div');
        header.className = 'command-palette-category';
        header.textContent = category;
        this.resultsContainer.appendChild(header);
      }
      
      // Commands
      commands.forEach(cmd => {
        const item = this.createCommandItem(cmd, index);
        this.resultsContainer.appendChild(item);
        index++;
      });
    }
  }
  
  groupByCategory(commands) {
    const grouped = {};
    
    commands.forEach(cmd => {
      const category = cmd.isRecent ? 'Recent' : cmd.category;
      if (!grouped[category]) {
        grouped[category] = [];
      }
      grouped[category].push(cmd);
    });
    
    // Order: Recent, Quick Actions, Advanced, Navigation, System
    const order = ['Recent', 'Quick Actions', 'Advanced', 'Navigation', 'System'];
    const ordered = {};
    order.forEach(cat => {
      if (grouped[cat]) {
        ordered[cat] = grouped[cat];
      }
    });
    
    return ordered;
  }
  
  createCommandItem(cmd, index) {
    const item = document.createElement('div');
    item.className = 'command-palette-item';
    item.setAttribute('role', 'option');
    item.setAttribute('data-index', index);
    
    if (index === this.selectedIndex) {
      item.classList.add('selected');
    }
    
    item.innerHTML = `
      <div class="command-palette-item-icon">${cmd.icon}</div>
      <div class="command-palette-item-content">
        <div class="command-palette-item-name">
          ${cmd.name}
          ${cmd.isRecent ? '<span class="command-palette-badge">Recent</span>' : ''}
        </div>
        <div class="command-palette-item-description">${cmd.description}</div>
      </div>
    `;
    
    item.addEventListener('click', () => {
      this.selectedIndex = index;
      this.executeSelected();
    });
    
    item.addEventListener('mouseenter', () => {
      this.selectedIndex = index;
      this.render();
    });
    
    return item;
  }
  
  selectNext() {
    if (this.filteredCommands.length === 0) return;
    this.selectedIndex = (this.selectedIndex + 1) % this.filteredCommands.length;
    this.render();
    this.scrollToSelected();
  }
  
  selectPrevious() {
    if (this.filteredCommands.length === 0) return;
    this.selectedIndex = this.selectedIndex === 0 
      ? this.filteredCommands.length - 1 
      : this.selectedIndex - 1;
    this.render();
    this.scrollToSelected();
  }
  
  scrollToSelected() {
    const selected = this.resultsContainer.querySelector('.command-palette-item.selected');
    if (selected) {
      selected.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    }
  }
  
  executeSelected() {
    if (this.filteredCommands.length === 0) return;
    
    const command = this.filteredCommands[this.selectedIndex];
    if (command && command.action) {
      // Save to recent actions
      this.saveRecentAction(command.id);
      
      // Execute
      command.action();
      
      // Close palette
      this.close();
    }
  }
  
  saveRecentAction(commandId) {
    // Add to front, remove duplicates, keep last 5
    this.recentActions = [
      commandId,
      ...this.recentActions.filter(id => id !== commandId)
    ].slice(0, 5);
    
    // Persist
    chrome.storage.local.set({ recentActions: this.recentActions });
  }
  
  loadRecentActions() {
    // Load from storage (will be async, but we handle it)
    chrome.storage.local.get(['recentActions'], (result) => {
      if (result.recentActions) {
        this.recentActions = result.recentActions;
      }
    });
    return [];
  }
  
  open() {
    if (this.isOpen) return;
    
    this.isOpen = true;
    this.backdrop.classList.add('visible');
    this.container.classList.add('visible');
    
    // Reset state
    this.searchInput.value = '';
    this.handleSearch('');
    
    // Focus input
    setTimeout(() => this.searchInput.focus(), 50);
  }
  
  close() {
    if (!this.isOpen) return;
    
    this.isOpen = false;
    this.backdrop.classList.remove('visible');
    this.container.classList.remove('visible');
  }
  
  toggle() {
    if (this.isOpen) {
      this.close();
    } else {
      this.open();
    }
  }
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    window.commandPalette = new CommandPalette();
  });
} else {
  window.commandPalette = new CommandPalette();
}

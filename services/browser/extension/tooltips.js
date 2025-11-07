/**
 * Tooltip System - Contextual help for UI elements
 * Provides helpful tips and examples for features
 */

class TooltipManager {
  constructor() {
    this.tooltips = new Map();
    this.activeTooltip = null;
    this.createTooltipElement();
    this.registerTooltips();
    this.attachEventListeners();
  }
  
  createTooltipElement() {
    this.tooltip = document.createElement('div');
    this.tooltip.className = 'tooltip';
    this.tooltip.setAttribute('role', 'tooltip');
    document.body.appendChild(this.tooltip);
  }
  
  registerTooltips() {
    // Define tooltips for each feature
    const tooltipData = {
      'run-ocr': {
        title: 'Extract Text',
        content: 'Extract text, tables, and images from documents and web pages',
        example: 'Try this on: PDFs, screenshots, scanned documents',
        shortcut: null
      },
      'run-sql': {
        title: 'Query Data',
        content: 'Run SQL queries on your connected database',
        example: 'Example: SELECT * FROM users LIMIT 10',
        shortcut: null
      },
      'run-telemetry': {
        title: 'View Telemetry',
        content: 'Monitor system metrics, logs, and performance data',
        example: 'See recent API calls, errors, and response times',
        shortcut: null
      },
      'open-browser': {
        title: 'Browser Shell',
        content: 'Open the full-featured Electron browser interface',
        example: 'Access advanced features like data visualization',
        shortcut: null
      },
      'run-agentflow': {
        title: 'AgentFlow',
        content: 'Execute automated workflows and agent chains',
        example: 'Run multi-step automation tasks',
        shortcut: null
      },
      'run-search': {
        title: 'OpenSearch',
        content: 'Query your OpenSearch/Elasticsearch indices',
        example: 'Full-text search across your data',
        shortcut: null
      },
      'redis-set': {
        title: 'Redis Set',
        content: 'Store key-value pairs in Redis cache',
        example: 'Cache frequently accessed data',
        shortcut: null
      },
      'redis-get': {
        title: 'Redis Get',
        content: 'Retrieve values from Redis cache by key',
        example: 'Fast lookup of cached data',
        shortcut: null
      },
      'chat-send': {
        title: 'LocalAI Chat',
        content: 'Chat with your local AI models',
        example: 'Ask questions, get summaries, generate content',
        shortcut: 'Cmd/Ctrl+Enter'
      },
      'toggle-advanced': {
        title: 'Advanced Tools',
        content: 'Show or hide advanced features',
        example: 'Toggle to access Redis, OpenSearch, AgentFlow',
        shortcut: null
      },
      'settings-link': {
        title: 'Settings',
        content: 'Configure gateway URL and browser shell settings',
        example: 'Test your gateway connection here',
        shortcut: null
      },
      'help-link': {
        title: 'Help & Documentation',
        content: 'Access guides, tutorials, and API documentation',
        example: 'Learn how to use all features',
        shortcut: null
      }
    };
    
    // Create tooltip instances
    Object.entries(tooltipData).forEach(([id, data]) => {
      this.tooltips.set(id, data);
    });
  }
  
  attachEventListeners() {
    // Add listeners to all elements with tooltips
    this.tooltips.forEach((data, id) => {
      const element = document.getElementById(id);
      if (element) {
        element.addEventListener('mouseenter', (e) => this.show(id, element));
        element.addEventListener('mouseleave', () => this.hide());
        element.addEventListener('focus', (e) => this.show(id, element));
        element.addEventListener('blur', () => this.hide());
      }
    });
    
    // Hide on scroll
    window.addEventListener('scroll', () => this.hide(), true);
  }
  
  show(id, element) {
    const data = this.tooltips.get(id);
    if (!data) return;
    
    // Build tooltip content
    let content = `<div class="tooltip-title">${data.title}</div>`;
    content += `<div class="tooltip-content">${data.content}</div>`;
    
    if (data.example) {
      content += `<div class="tooltip-example">💡 ${data.example}</div>`;
    }
    
    if (data.shortcut) {
      content += `<div class="tooltip-shortcut"><kbd>${data.shortcut}</kbd></div>`;
    }
    
    this.tooltip.innerHTML = content;
    this.tooltip.classList.add('visible');
    this.activeTooltip = id;
    
    // Position tooltip
    this.position(element);
  }
  
  position(element) {
    const rect = element.getBoundingClientRect();
    const tooltipRect = this.tooltip.getBoundingClientRect();
    
    // Default: below element, centered
    let top = rect.bottom + 8;
    let left = rect.left + (rect.width / 2) - (tooltipRect.width / 2);
    
    // Check if tooltip would go off-screen
    const viewportWidth = window.innerWidth;
    const viewportHeight = window.innerHeight;
    
    // Adjust horizontal position
    if (left < 8) {
      left = 8;
    } else if (left + tooltipRect.width > viewportWidth - 8) {
      left = viewportWidth - tooltipRect.width - 8;
    }
    
    // Adjust vertical position (flip to top if not enough space below)
    if (top + tooltipRect.height > viewportHeight - 8) {
      top = rect.top - tooltipRect.height - 8;
      this.tooltip.classList.add('top');
    } else {
      this.tooltip.classList.remove('top');
    }
    
    this.tooltip.style.top = `${top}px`;
    this.tooltip.style.left = `${left}px`;
  }
  
  hide() {
    this.tooltip.classList.remove('visible');
    this.activeTooltip = null;
  }
}

// Help icon component - adds (?) icon to elements
class HelpIconManager {
  constructor() {
    this.helpData = {
      'connection-status': {
        title: 'Connection Status',
        content: 'Shows whether the gateway is online and reachable. The extension checks the connection every 30 seconds.',
        actions: [
          'Green dot = Connected and ready',
          'Red dot = Gateway offline',
          'Blue dot = Checking connection'
        ]
      },
      'chat-section': {
        title: 'LocalAI Chat',
        content: 'Send messages to your local AI models. Press Cmd/Ctrl+Enter to send.',
        actions: [
          'Leave model blank for default',
          'Supports OpenAI-compatible models',
          'Chat history is not persisted'
        ]
      }
    };
    
    this.createHelpIcons();
  }
  
  createHelpIcons() {
    Object.entries(this.helpData).forEach(([id, data]) => {
      const element = document.getElementById(id);
      if (!element) return;
      
      const helpIcon = document.createElement('button');
      helpIcon.className = 'help-icon';
      helpIcon.innerHTML = '?';
      helpIcon.setAttribute('aria-label', `Help for ${data.title}`);
      helpIcon.setAttribute('type', 'button');
      
      helpIcon.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        this.showHelpDialog(data);
      });
      
      // Position icon relative to element
      element.style.position = 'relative';
      element.appendChild(helpIcon);
    });
  }
  
  showHelpDialog(data) {
    // Create dialog
    const dialog = document.createElement('div');
    dialog.className = 'help-dialog';
    dialog.setAttribute('role', 'dialog');
    dialog.setAttribute('aria-modal', 'true');
    
    let content = `
      <div class="help-dialog-header">
        <h3>${data.title}</h3>
        <button class="help-dialog-close" aria-label="Close help">&times;</button>
      </div>
      <div class="help-dialog-body">
        <p>${data.content}</p>
    `;
    
    if (data.actions && data.actions.length > 0) {
      content += '<ul>';
      data.actions.forEach(action => {
        content += `<li>${action}</li>`;
      });
      content += '</ul>';
    }
    
    content += '</div>';
    dialog.innerHTML = content;
    
    // Add backdrop
    const backdrop = document.createElement('div');
    backdrop.className = 'help-dialog-backdrop';
    
    document.body.appendChild(backdrop);
    document.body.appendChild(dialog);
    
    // Close handlers
    const close = () => {
      dialog.remove();
      backdrop.remove();
    };
    
    dialog.querySelector('.help-dialog-close').addEventListener('click', close);
    backdrop.addEventListener('click', close);
    
    // Escape key
    const escHandler = (e) => {
      if (e.key === 'Escape') {
        close();
        document.removeEventListener('keydown', escHandler);
      }
    };
    document.addEventListener('keydown', escHandler);
  }
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    window.tooltipManager = new TooltipManager();
    window.helpIconManager = new HelpIconManager();
  });
} else {
  window.tooltipManager = new TooltipManager();
  window.helpIconManager = new HelpIconManager();
}

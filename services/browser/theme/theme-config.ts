/**
 * Unified Theme Configuration for aModels
 * Provides consistent theming across all UI services
 */

export interface ThemeColors {
  primary: string;
  secondary: string;
  success: string;
  warning: string;
  error: string;
  info: string;
  background: {
    default: string;
    paper: string;
    dark: string;
  };
  text: {
    primary: string;
    secondary: string;
    disabled: string;
  };
}

export interface ThemeConfig {
  colors: ThemeColors;
  typography: {
    fontFamily: string;
    fontSize: {
      small: string;
      base: string;
      large: string;
      xlarge: string;
    };
  };
  spacing: {
    unit: number;
    small: string;
    medium: string;
    large: string;
  };
  borderRadius: {
    small: string;
    medium: string;
    large: string;
  };
  shadows: {
    small: string;
    medium: string;
    large: string;
  };
}

// Main aModels theme configuration
export const aModelsTheme: ThemeConfig = {
  colors: {
    primary: '#667eea',
    secondary: '#764ba2',
    success: '#10b981',
    warning: '#f59e0b',
    error: '#ef4444',
    info: '#3b82f6',
    background: {
      default: '#f8fafc',
      paper: '#ffffff',
      dark: '#1e293b'
    },
    text: {
      primary: '#1e293b',
      secondary: '#64748b',
      disabled: '#cbd5e1'
    }
  },
  typography: {
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
    fontSize: {
      small: '0.875rem',
      base: '1rem',
      large: '1.25rem',
      xlarge: '1.5rem'
    }
  },
  spacing: {
    unit: 8,
    small: '0.5rem',
    medium: '1rem',
    large: '2rem'
  },
  borderRadius: {
    small: '4px',
    medium: '8px',
    large: '12px'
  },
  shadows: {
    small: '0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)',
    medium: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
    large: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)'
  }
};

// Material-UI theme configuration
export const getMuiTheme = () => ({
  palette: {
    primary: {
      main: aModelsTheme.colors.primary,
      contrastText: '#ffffff'
    },
    secondary: {
      main: aModelsTheme.colors.secondary,
      contrastText: '#ffffff'
    },
    success: {
      main: aModelsTheme.colors.success
    },
    warning: {
      main: aModelsTheme.colors.warning
    },
    error: {
      main: aModelsTheme.colors.error
    },
    info: {
      main: aModelsTheme.colors.info
    },
    background: {
      default: aModelsTheme.colors.background.default,
      paper: aModelsTheme.colors.background.paper
    },
    text: {
      primary: aModelsTheme.colors.text.primary,
      secondary: aModelsTheme.colors.text.secondary,
      disabled: aModelsTheme.colors.text.disabled
    }
  },
  typography: {
    fontFamily: aModelsTheme.typography.fontFamily,
    fontSize: 14,
    h1: { fontSize: '2.5rem', fontWeight: 600 },
    h2: { fontSize: '2rem', fontWeight: 600 },
    h3: { fontSize: '1.75rem', fontWeight: 600 },
    h4: { fontSize: '1.5rem', fontWeight: 600 },
    h5: { fontSize: '1.25rem', fontWeight: 600 },
    h6: { fontSize: '1rem', fontWeight: 600 },
    body1: { fontSize: '1rem' },
    body2: { fontSize: '0.875rem' }
  },
  shape: {
    borderRadius: 8
  },
  spacing: 8
});

// CSS variables export for non-React components
export const getCSSVariables = (): string => {
  return `
    :root {
      --color-primary: ${aModelsTheme.colors.primary};
      --color-secondary: ${aModelsTheme.colors.secondary};
      --color-success: ${aModelsTheme.colors.success};
      --color-warning: ${aModelsTheme.colors.warning};
      --color-error: ${aModelsTheme.colors.error};
      --color-info: ${aModelsTheme.colors.info};
      
      --bg-default: ${aModelsTheme.colors.background.default};
      --bg-paper: ${aModelsTheme.colors.background.paper};
      --bg-dark: ${aModelsTheme.colors.background.dark};
      
      --text-primary: ${aModelsTheme.colors.text.primary};
      --text-secondary: ${aModelsTheme.colors.text.secondary};
      --text-disabled: ${aModelsTheme.colors.text.disabled};
      
      --font-family: ${aModelsTheme.typography.fontFamily};
      --font-size-small: ${aModelsTheme.typography.fontSize.small};
      --font-size-base: ${aModelsTheme.typography.fontSize.base};
      --font-size-large: ${aModelsTheme.typography.fontSize.large};
      --font-size-xlarge: ${aModelsTheme.typography.fontSize.xlarge};
      
      --spacing-small: ${aModelsTheme.spacing.small};
      --spacing-medium: ${aModelsTheme.spacing.medium};
      --spacing-large: ${aModelsTheme.spacing.large};
      
      --radius-small: ${aModelsTheme.borderRadius.small};
      --radius-medium: ${aModelsTheme.borderRadius.medium};
      --radius-large: ${aModelsTheme.borderRadius.large};
      
      --shadow-small: ${aModelsTheme.shadows.small};
      --shadow-medium: ${aModelsTheme.shadows.medium};
      --shadow-large: ${aModelsTheme.shadows.large};
    }
  `;
};

export default aModelsTheme;

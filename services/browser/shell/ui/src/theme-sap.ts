import { createTheme, alpha } from '@mui/material/styles';

/**
 * SAP Horizon Theme for Material-UI
 * Aligned with SAP Workzone, Datasphere, and Analytics Cloud
 * Based on SAP Fiori 3 and SAP Horizon Design Guidelines
 */

const sapHorizonTheme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#0a6ed1',      // SAP Blue
      light: '#3f9ddb',
      dark: '#0854a0',
      contrastText: '#ffffff',
    },
    secondary: {
      main: '#354a5f',      // SAP Shell Blue
      light: '#475e75',
      dark: '#1d2d3e',
      contrastText: '#ffffff',
    },
    error: {
      main: '#bb0000',      // SAP Negative
      light: '#e00',
      dark: '#a20000',
      contrastText: '#ffffff',
    },
    warning: {
      main: '#e9730c',      // SAP Critical
      light: '#f09d3c',
      dark: '#d7660a',
      contrastText: '#ffffff',
    },
    success: {
      main: '#107e3e',      // SAP Positive
      light: '#2da955',
      dark: '#0d6733',
      contrastText: '#ffffff',
    },
    info: {
      main: '#0a6ed1',      // SAP Informative
      light: '#3f9ddb',
      dark: '#0854a0',
      contrastText: '#ffffff',
    },
    background: {
      default: '#f5f5f5',   // SAP Gray-2
      paper: '#ffffff',
    },
    text: {
      primary: '#32363a',   // SAP Gray-9
      secondary: '#6a6d70', // SAP Gray-7
      disabled: '#bfbfbf',  // SAP Gray-5
    },
    divider: '#d9d9d9',     // SAP Gray-4
    action: {
      active: '#0a6ed1',
      hover: alpha('#0a6ed1', 0.08),
      selected: alpha('#0a6ed1', 0.12),
      disabled: '#bfbfbf',
      disabledBackground: '#ededed',
    },
  },
  typography: {
    fontFamily: '"72", "72full", Arial, Helvetica, sans-serif',
    fontSize: 14,
    fontWeightLight: 300,
    fontWeightRegular: 400,
    fontWeightMedium: 600,
    fontWeightBold: 700,
    h1: {
      fontSize: '1.75rem',    // 28px
      fontWeight: 700,
      lineHeight: 1.2,
      letterSpacing: '-0.01em',
    },
    h2: {
      fontSize: '1.375rem',   // 22px
      fontWeight: 700,
      lineHeight: 1.2,
      letterSpacing: '-0.005em',
    },
    h3: {
      fontSize: '1.125rem',   // 18px
      fontWeight: 600,
      lineHeight: 1.2,
    },
    h4: {
      fontSize: '1rem',       // 16px
      fontWeight: 600,
      lineHeight: 1.4,
    },
    h5: {
      fontSize: '0.875rem',   // 14px
      fontWeight: 600,
      lineHeight: 1.4,
    },
    h6: {
      fontSize: '0.75rem',    // 12px
      fontWeight: 600,
      lineHeight: 1.4,
      textTransform: 'uppercase',
      letterSpacing: '0.05em',
    },
    body1: {
      fontSize: '0.875rem',   // 14px
      lineHeight: 1.6,
    },
    body2: {
      fontSize: '0.75rem',    // 12px
      lineHeight: 1.5,
    },
    button: {
      fontSize: '0.875rem',
      fontWeight: 400,
      textTransform: 'none',  // SAP doesn't uppercase buttons
      letterSpacing: 'normal',
    },
    caption: {
      fontSize: '0.75rem',
      lineHeight: 1.4,
      color: '#6a6d70',
    },
    overline: {
      fontSize: '0.75rem',
      fontWeight: 600,
      lineHeight: 1.4,
      textTransform: 'uppercase',
      letterSpacing: '0.08em',
    },
  },
  shape: {
    borderRadius: 4,        // SAP uses subtle rounding
  },
  spacing: 8,               // SAP 8pt grid system
  shadows: [
    'none',
    '0 0.0625rem 0.1875rem 0 rgba(0, 0, 0, 0.15)',
    '0 0.125rem 0.5rem 0 rgba(0, 0, 0, 0.15)',
    '0 0.25rem 1rem 0 rgba(0, 0, 0, 0.15)',
    '0 0.5rem 2rem 0 rgba(0, 0, 0, 0.15)',
    '0 0.5rem 2rem 0 rgba(0, 0, 0, 0.15)',
    '0 0.5rem 2rem 0 rgba(0, 0, 0, 0.15)',
    '0 0.5rem 2rem 0 rgba(0, 0, 0, 0.15)',
    '0 0.5rem 2rem 0 rgba(0, 0, 0, 0.15)',
    '0 0.5rem 2rem 0 rgba(0, 0, 0, 0.15)',
    '0 0.5rem 2rem 0 rgba(0, 0, 0, 0.15)',
    '0 0.5rem 2rem 0 rgba(0, 0, 0, 0.15)',
    '0 0.5rem 2rem 0 rgba(0, 0, 0, 0.15)',
    '0 0.5rem 2rem 0 rgba(0, 0, 0, 0.15)',
    '0 0.5rem 2rem 0 rgba(0, 0, 0, 0.15)',
    '0 0.5rem 2rem 0 rgba(0, 0, 0, 0.15)',
    '0 0.5rem 2rem 0 rgba(0, 0, 0, 0.15)',
    '0 0.5rem 2rem 0 rgba(0, 0, 0, 0.15)',
    '0 0.5rem 2rem 0 rgba(0, 0, 0, 0.15)',
    '0 0.5rem 2rem 0 rgba(0, 0, 0, 0.15)',
    '0 0.5rem 2rem 0 rgba(0, 0, 0, 0.15)',
    '0 0.5rem 2rem 0 rgba(0, 0, 0, 0.15)',
    '0 0.5rem 2rem 0 rgba(0, 0, 0, 0.15)',
    '0 0.5rem 2rem 0 rgba(0, 0, 0, 0.15)',
    '0 0.5rem 2rem 0 rgba(0, 0, 0, 0.15)',
  ],
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        body: {
          backgroundColor: '#f5f5f5',
          color: '#32363a',
          fontSize: '14px',
        },
        '@font-face': {
          fontFamily: '72',
          fontStyle: 'normal',
          fontDisplay: 'swap',
          fontWeight: 400,
          // Note: In production, load SAP 72 font from SAP CDN or local assets
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontWeight: 400,
          borderRadius: 4,
          padding: '8px 16px',
          boxShadow: 'none',
          '&:hover': {
            boxShadow: '0 0.0625rem 0.1875rem 0 rgba(0, 0, 0, 0.15)',
          },
        },
        contained: {
          boxShadow: 'none',
          '&:hover': {
            boxShadow: '0 0.0625rem 0.1875rem 0 rgba(0, 0, 0, 0.15)',
          },
          '&:active': {
            boxShadow: 'none',
          },
        },
        containedPrimary: {
          backgroundColor: '#0a6ed1',
          '&:hover': {
            backgroundColor: '#0854a0',
          },
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
          boxShadow: '0 0.0625rem 0.1875rem 0 rgba(0, 0, 0, 0.15)',
        },
        elevation1: {
          boxShadow: '0 0.0625rem 0.1875rem 0 rgba(0, 0, 0, 0.15)',
        },
        elevation2: {
          boxShadow: '0 0.125rem 0.5rem 0 rgba(0, 0, 0, 0.15)',
        },
        elevation3: {
          boxShadow: '0 0.25rem 1rem 0 rgba(0, 0, 0, 0.15)',
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          boxShadow: '0 0.0625rem 0.1875rem 0 rgba(0, 0, 0, 0.15)',
        },
      },
    },
    MuiCardHeader: {
      styleOverrides: {
        root: {
          borderBottom: '1px solid #d9d9d9',
          padding: '16px',
        },
        title: {
          fontSize: '1.125rem',
          fontWeight: 700,
          color: '#32363a',
        },
        subheader: {
          fontSize: '0.875rem',
          color: '#6a6d70',
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            '& fieldset': {
              borderColor: '#d9d9d9',
            },
            '&:hover fieldset': {
              borderColor: '#bfbfbf',
            },
            '&.Mui-focused fieldset': {
              borderColor: '#0a6ed1',
              borderWidth: 1,
            },
          },
        },
      },
    },
    MuiOutlinedInput: {
      styleOverrides: {
        root: {
          borderRadius: 4,
          '&:hover .MuiOutlinedInput-notchedOutline': {
            borderColor: '#bfbfbf',
          },
          '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
            borderColor: '#0a6ed1',
            borderWidth: 1,
          },
        },
      },
    },
    MuiAppBar: {
      styleOverrides: {
        root: {
          backgroundColor: '#354a5f',
          boxShadow: '0 0.0625rem 0.1875rem 0 rgba(0, 0, 0, 0.15)',
        },
      },
    },
    MuiToolbar: {
      styleOverrides: {
        root: {
          minHeight: '44px !important',  // SAP Shell Bar height
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          borderRadius: 4,
          height: 24,
          fontSize: '0.75rem',
        },
      },
    },
    MuiAlert: {
      styleOverrides: {
        root: {
          borderRadius: 4,
          borderLeft: '4px solid',
        },
        standardSuccess: {
          backgroundColor: '#f1fdf6',
          borderColor: '#107e3e',
          color: '#32363a',
        },
        standardError: {
          backgroundColor: '#ffebeb',
          borderColor: '#bb0000',
          color: '#32363a',
        },
        standardWarning: {
          backgroundColor: '#fef7f1',
          borderColor: '#e9730c',
          color: '#32363a',
        },
        standardInfo: {
          backgroundColor: '#eef4fb',
          borderColor: '#0a6ed1',
          color: '#32363a',
        },
      },
    },
    MuiTableCell: {
      styleOverrides: {
        head: {
          backgroundColor: '#f5f5f5',
          color: '#32363a',
          fontWeight: 600,
          fontSize: '0.875rem',
        },
        body: {
          fontSize: '0.875rem',
        },
      },
    },
    MuiDrawer: {
      styleOverrides: {
        paper: {
          backgroundColor: '#fafafa',
          borderRight: '1px solid #d9d9d9',
        },
      },
    },
    MuiListItemButton: {
      styleOverrides: {
        root: {
          '&:hover': {
            backgroundColor: alpha('#0a6ed1', 0.08),
          },
          '&.Mui-selected': {
            backgroundColor: alpha('#0a6ed1', 0.12),
            '&:hover': {
              backgroundColor: alpha('#0a6ed1', 0.16),
            },
          },
        },
      },
    },
  },
});

export default sapHorizonTheme;

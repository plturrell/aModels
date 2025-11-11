import { createTheme, alpha } from '@mui/material/styles';

/**
 * Agent's Workbench - Professional Dark Theme
 * Inspired by Figma, Retool, and professional IDEs.
 */
const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#00A3FF', // A brighter, more energetic blue
    },
    background: {
      default: '#121212', // Standard dark theme background
      paper: '#1E1E1E',   // Elevated surfaces, like cards and panels
    },
    text: {
      primary: '#EAEAEA',
      secondary: '#A0A0A0',
    },
    divider: alpha('#EAEAEA', 0.12),
  },
  typography: {
    fontFamily: ['-apple-system', 'BlinkMacSystemFont', '"Inter"', 'sans-serif'].join(','),
    h1: { fontSize: '2.5rem', fontWeight: 700, letterSpacing: '-0.02em' },
    h2: { fontSize: '2rem', fontWeight: 700, letterSpacing: '-0.015em' },
    h3: { fontSize: '1.75rem', fontWeight: 600, letterSpacing: '-0.01em' },
    h4: { fontSize: '1.5rem', fontWeight: 600, letterSpacing: '-0.005em' },
    h5: { fontSize: '1.25rem', fontWeight: 600 },
    h6: { fontSize: '1rem', fontWeight: 600 },
    body1: { fontSize: '1rem', lineHeight: 1.6 },
    body2: { fontSize: '0.875rem', lineHeight: 1.5, color: '#A0A0A0' },
    caption: { fontSize: '0.75rem', color: '#A0A0A0' },
    overline: { fontSize: '0.75rem', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.08em' },
  },
  shape: {
    borderRadius: 8, // Slightly sharper for a more 'pro' feel
  },
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        body: {
          backgroundColor: '#121212',
          color: '#EAEAEA',
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none', // Remove gradients from Paper components
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontWeight: 600,
        },
        contained: {
          boxShadow: 'none',
          '&:hover': {
            boxShadow: 'none',
          },
        },
      },
    },
  },
});

export default theme;

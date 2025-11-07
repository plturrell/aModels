# Perplexity Dashboard

Beautiful, interactive dashboard for exploring Perplexity processing results — designed with the **Jobs & Ive lens**.

## Design Philosophy

- **Simplicity**: Clean, focused interfaces
- **Beauty**: Elegant typography, generous whitespace, purposeful colors
- **Intuition**: Zero learning curve
- **Delight**: Smooth animations, beautiful interactions

## Getting Started

### Prerequisites

- Node.js >= 18
- npm or yarn

### Installation

```bash
# Install dependencies
npm install

# Or if using local Observable services
npm install --workspace=../../framework
npm install --workspace=../../plot
npm install --workspace=../../runtime
npm install --workspace=../../stdlib
```

### Development

```bash
# Start development server
npm run dev

# Build for production
npm run build

# Deploy
npm run deploy
```

## Project Structure

```
dashboard/
├── src/                    # Source files (markdown pages)
│   ├── index.md          # Landing page
│   ├── processing.md     # Processing dashboard
│   ├── results.md        # Results dashboard
│   ├── analytics.md      # Analytics dashboard
│   ├── graph.md          # Knowledge graph dashboard
│   └── query.md          # Query dashboard
├── data/
│   └── loaders/          # Data loaders for API integration
│       ├── processing.js
│       ├── results.js
│       ├── intelligence.js
│       └── analytics.js
├── static/                # Static assets
│   ├── css/
│   │   └── styles.css    # Design system
│   └── images/
├── observable.config.js   # Framework configuration
└── package.json
```

## API Integration

The dashboard connects to the Perplexity API. Set the base URL:

```bash
export PERPLEXITY_API_BASE=http://localhost:8080
```

## Design System

See `src/styles.css` for the complete design system:
- Color palette (iOS-inspired)
- Typography scale
- Spacing system
- Component patterns
- Animation principles

## Resources

- [Observable Framework Docs](https://observablehq.com/framework/)
- [Observable Plot Gallery](https://observablehq.com/@observablehq/plot-gallery)
- [Design System Documentation](../../../docs/PERPLEXITY_DESIGN_SYSTEM.md)


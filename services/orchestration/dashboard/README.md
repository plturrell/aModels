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
│   ├── index.md           # Landing page
│   ├── processing.md      # Processing dashboard
│   ├── results.md         # Results dashboard
│   ├── analytics.md       # Analytics dashboard
│   ├── graph.md           # Knowledge graph dashboard
│   ├── query.md           # Query dashboard
│   ├── components/        # Reusable components
│   │   ├── export.js     # Export utilities
│   │   └── emptyState.js # Empty state components
│   └── styles.css         # Design system
├── data/
│   └── loaders/          # Data loaders for API integration
│       ├── processing.js
│       ├── results.js
│       ├── intelligence.js
│       ├── analytics.js
│       └── graph.js
├── observable.config.js   # Framework configuration
└── package.json
```

## API Integration

The dashboard connects to the Perplexity API. Set the base URL:

```bash
export PERPLEXITY_API_BASE=http://localhost:8000
```

Or create a `.env` file:

```
PERPLEXITY_API_BASE=http://localhost:8000
```

## Deep Linking

All dashboards support deep linking via URL parameters:

- **Processing**: `/processing?request_id=req_123`
- **Results**: `/results?request_id=req_123`
- **Graph**: `/graph?request_id=req_123`
- **Query**: `/query?request_id=req_123`

## Features

### Phase 1: Foundation ✅
- Dashboard structure
- Design system
- API integration layer

### Phase 2: Core Visualizations ✅
- Processing status dashboard
- Results exploration dashboard
- Analytics dashboard
- 10+ interactive visualizations

### Phase 3: Advanced Features ✅
- Knowledge Graph dashboard
- Query dashboard
- Real-time auto-refresh
- Export functionality (PNG/SVG/JSON/CSV)

### Phase 4: Integration & Polish ✅
- Deep linking support
- Beautiful empty states
- Error handling
- Performance optimization

## Design System

See `src/styles.css` for the complete design system:
- Color palette (iOS-inspired)
- Typography scale
- Spacing system
- Component patterns

## API Endpoints

The dashboard uses these Perplexity API endpoints:

- `GET /api/perplexity/status/{request_id}` - Processing status
- `GET /api/perplexity/results/{request_id}` - Results data
- `GET /api/perplexity/results/{request_id}/intelligence` - Intelligence data
- `GET /api/perplexity/history` - Request history
- `POST /api/perplexity/search` - Search documents
- `GET /api/perplexity/graph/{request_id}/relationships` - Graph relationships
- `POST /api/perplexity/graph/{request_id}/query` - Graph queries

## Export Functionality

Export charts and data:

```javascript
import {exportJSON, exportCSV, exportChartPNG, exportChartSVG} from "./components/export.js";

// Export data
exportJSON(data, "intelligence.json");
exportCSV(documents, "documents.csv");

// Export charts (requires chart element)
exportChartPNG(chartElement, "chart.png");
exportChartSVG(chartElement, "chart.svg");
```

## Real-time Updates

The processing dashboard automatically refreshes every 2 seconds for active requests, stopping when processing completes or fails.

## Contributing

When adding new dashboards or features:

1. Follow the Jobs & Ive design lens
2. Use the design system from `styles.css`
3. Create reusable components in `components/`
4. Add data loaders in `data/loaders/`
5. Support deep linking via URL parameters
6. Include beautiful empty states

## License

Part of the aModels project.

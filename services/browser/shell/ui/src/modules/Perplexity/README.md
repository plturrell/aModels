# Perplexity Dashboard Module

This module embeds the Observable Framework-based Perplexity Dashboard into the Browser Shell, providing a unified interface for visualizing Perplexity processing results.

## Features

- **Embedded Dashboard**: Full Observable Framework dashboard in Browser Shell
- **Unified Navigation**: Access from Browser Shell sidebar
- **Shared Configuration**: Uses same API endpoints as other modules
- **Beautiful Design**: Jobs & Ive lens design principles

## Dashboard Pages

The embedded dashboard includes:

1. **Processing Dashboard** - Real-time processing status, progress, statistics
2. **Results Dashboard** - Document exploration, relationships, patterns
3. **Analytics Dashboard** - Trends, patterns, domain distributions
4. **Knowledge Graph Dashboard** - Interactive graph visualization (Phase 3)
5. **Query Dashboard** - Search interface (Phase 3)

## Configuration

Set the dashboard URL via environment variable:

```bash
# .env file
VITE_DASHBOARD_URL=http://localhost:3000
```

Or when running the shell:

```bash
VITE_DASHBOARD_URL=http://localhost:3000 npm start
```

## Prerequisites

1. **Observable Framework Dashboard Running**:
   ```bash
   cd services/orchestration/dashboard
   npm run dev
   ```

2. **Perplexity API Running**:
   ```bash
   # Make sure orchestration service is running on port 8080
   # or update PERPLEXITY_API_BASE in dashboard/.env
   ```

## Integration Details

- **Module Location**: `src/modules/Perplexity/`
- **Component**: `PerplexityModule.tsx`
- **Navigation**: Added to `NavPanel.tsx`
- **Store**: Updated `ShellModuleId` type
- **App**: Registered in `App.tsx`

## Design

The module follows the same design principles as the dashboard:
- **Simplicity**: Clean iframe embedding
- **Beauty**: Smooth loading transitions
- **Intuition**: Clear error messages
- **Delight**: Seamless integration

## Troubleshooting

### Dashboard not loading

1. Check if Observable Framework server is running:
   ```bash
   curl http://localhost:3000
   ```

2. Verify `VITE_DASHBOARD_URL` is set correctly

3. Check browser console for CORS errors

### API connection errors

1. Verify Perplexity API is running on port 8080
2. Check dashboard `.env` file for `PERPLEXITY_API_BASE`
3. Ensure CORS is enabled on the API

## Next Steps

- [ ] Add direct API integration (bypass iframe)
- [ ] Share authentication tokens
- [ ] Add dashboard controls in Browser Shell
- [ ] Implement real-time updates via WebSocket


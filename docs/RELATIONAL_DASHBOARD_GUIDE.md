# Relational Dashboard Guide

## Overview

The Relational Dashboard provides beautiful, interactive visualizations for exploring relational table processing results. Built with Observable Framework and Plot, following the Jobs & Ive design philosophy.

## Dashboard Pages

### 1. Relational Landing Page (`/relational`)

The main entry point for the relational dashboard, providing:
- Quick navigation to all dashboards
- Overview of features
- Design philosophy

### 2. Processing Dashboard (`/relational-processing`)

Monitor table processing in real-time:

**Features:**
- Real-time status updates (auto-refresh every 2 seconds)
- Progress tracking with percentage and ETA
- Table statistics (processed, succeeded, failed)
- Step-by-step progress timeline
- Error reporting with detailed information

**Usage:**
- Enter a request ID or use deep linking: `?request_id=xxx`
- View processing status and progress
- Monitor errors and warnings

### 3. Results Dashboard (`/relational-results`)

Explore processed tables and intelligence:

**Features:**
- Intelligence summary (domains, relationships, patterns, KG nodes)
- Domain distribution visualization
- Relationship network
- Pattern frequency charts
- Table list with intelligence data

**Usage:**
- Enter a request ID or use deep linking: `?request_id=xxx`
- Explore relationships and patterns
- Export results (JSON/CSV/PNG/SVG)

### 4. Analytics Dashboard (`/relational-analytics`)

Analyze trends and patterns:

**Features:**
- Analytics summary (total, completed, failed, success rate)
- Request volume over time
- Success rate trends
- Recent activity table

**Usage:**
- View overall processing performance
- Identify trends and patterns
- Monitor system health

### 5. Tables Library (`/relational-tables`)

Browse processed tables:

**Features:**
- Connection information
- Processed tables list
- Status indicators (processed/pending)

**Usage:**
- Browse processed tables
- View table details
- Track table processing status

## Deep Linking

All dashboards support deep linking via URL parameters:

- **Processing**: `/relational-processing?request_id=xxx`
- **Results**: `/relational-results?request_id=xxx`
- **Analytics**: `/relational-analytics`
- **Tables**: `/relational-tables`

## Export Functionality

Results can be exported in multiple formats:
- **JSON**: Full data export
- **CSV**: Tabular data export
- **PNG**: Chart images
- **SVG**: Vector chart images

## Design System

The dashboard follows the Jobs & Ive design lens:
- **Colors**: iOS-inspired palette (blues, grays, semantic colors)
- **Typography**: SF Pro-like fonts
- **Spacing**: 4px base unit system
- **Animations**: Smooth, purposeful transitions

## API Integration

All dashboards connect to the gateway API (`http://localhost:8000`):
- Processing status: `/api/relational/status/{request_id}`
- Results: `/api/relational/results/{request_id}`
- Intelligence: `/api/relational/results/{request_id}/intelligence`
- History: `/api/relational/history`
- Tables: Processed via `/api/relational/process`

## Error Handling

Dashboards gracefully handle errors:
- Connection errors: User-friendly error messages
- 404 errors: Empty states with helpful guidance
- API errors: Detailed error information

## Best Practices

1. **Use Deep Linking**: Share specific request views via URL
2. **Monitor Processing**: Use auto-refresh for real-time updates
3. **Export Results**: Save important results for analysis
4. **Explore Intelligence**: Discover relationships and patterns


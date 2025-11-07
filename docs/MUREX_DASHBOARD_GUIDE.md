# Murex Dashboard Guide

## Overview

The Murex Dashboard provides beautiful, interactive visualizations for exploring Murex trade processing results and ETL to SAP GL. Built with Observable Framework and Plot, following the Jobs & Ive design philosophy.

## Dashboard Pages

### 1. Murex Landing Page (`/murex`)

The main entry point for the Murex dashboard, providing:
- Quick navigation to all dashboards
- Overview of features
- Design philosophy

### 2. Processing Dashboard (`/murex-processing`)

Monitor trade processing in real-time:

**Features:**
- Real-time status updates (auto-refresh every 2 seconds)
- Progress tracking with percentage and ETA
- Trade statistics (processed, succeeded, failed)
- Step-by-step progress timeline
- Error reporting with detailed information

**Usage:**
- Enter a request ID or use deep linking: `?request_id=xxx`
- View processing status and progress
- Monitor errors and warnings

### 3. Results Dashboard (`/murex-results`)

Explore processed trades and intelligence:

**Features:**
- Intelligence summary (domains, relationships, patterns, KG nodes)
- Domain distribution visualization
- Relationship network
- Pattern frequency charts
- Trade list with intelligence data

**Usage:**
- Enter a request ID or use deep linking: `?request_id=xxx`
- Explore relationships and patterns
- Export results (JSON/CSV/PNG/SVG)

### 4. Analytics Dashboard (`/murex-analytics`)

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

### 5. ETL Dashboard (`/murex-etl`)

Monitor ETL transformations to SAP GL:

**Features:**
- ETL pipeline visualization
- Field mapping table
- Transformation steps
- SAP GL integration status

**Usage:**
- View ETL pipeline flow
- Understand field mappings
- Monitor transformation status

## Deep Linking

All dashboards support deep linking via URL parameters:

- **Processing**: `/murex-processing?request_id=xxx`
- **Results**: `/murex-results?request_id=xxx`
- **Analytics**: `/murex-analytics`
- **ETL**: `/murex-etl`

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
- Processing status: `/api/murex/status/{request_id}`
- Results: `/api/murex/results/{request_id}`
- Intelligence: `/api/murex/results/{request_id}/intelligence`
- History: `/api/murex/history`
- Process: `/api/murex/process`

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
5. **Monitor ETL**: Track SAP GL transformations


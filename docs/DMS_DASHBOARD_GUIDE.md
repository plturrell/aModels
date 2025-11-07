# DMS Dashboard Guide

## Overview

The DMS Dashboard provides beautiful, interactive visualizations for exploring document processing results. Built with Observable Framework and Plot, following the Jobs & Ive design philosophy.

## Dashboard Pages

### 1. DMS Landing Page (`/dms`)

The main entry point for the DMS dashboard, providing:
- Quick navigation to all dashboards
- Overview of features
- Design philosophy

### 2. Processing Dashboard (`/dms-processing`)

Monitor document processing in real-time:

**Features:**
- Real-time status updates (auto-refresh every 2 seconds)
- Progress tracking with percentage and ETA
- Document statistics (processed, succeeded, failed)
- Step-by-step progress timeline
- Error reporting with detailed information

**Usage:**
- Enter a request ID or use deep linking: `?request_id=xxx`
- View processing status and progress
- Monitor errors and warnings

### 3. Results Dashboard (`/dms-results`)

Explore processed documents and intelligence:

**Features:**
- Intelligence summary (domains, relationships, patterns, KG nodes)
- Domain distribution visualization
- Relationship network
- Pattern frequency charts
- Document list with intelligence data

**Usage:**
- Enter a request ID or use deep linking: `?request_id=xxx`
- Explore relationships and patterns
- Export results (JSON/CSV/PNG/SVG)

### 4. Analytics Dashboard (`/dms-analytics`)

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

### 5. Documents Library (`/dms-documents`)

Browse uploaded documents:

**Features:**
- Document list with metadata
- Status indicators (processed/pending)
- Document status distribution
- Documents over time chart

**Usage:**
- Browse all uploaded documents
- View document details
- Track document processing status

## Deep Linking

All dashboards support deep linking via URL parameters:

- **Processing**: `/dms-processing?request_id=xxx`
- **Results**: `/dms-results?request_id=xxx`
- **Analytics**: `/dms-analytics`
- **Documents**: `/dms-documents`

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
- Processing status: `/api/dms/status/{request_id}`
- Results: `/api/dms/results/{request_id}`
- Intelligence: `/api/dms/results/{request_id}/intelligence`
- History: `/api/dms/history`
- Documents: `/documents` (via DMS service)

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


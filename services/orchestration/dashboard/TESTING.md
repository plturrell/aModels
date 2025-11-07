# Phase 2 Testing Guide

## Overview

This guide helps you test the Phase 2 dashboards (Processing, Results, Analytics) to ensure everything works correctly.

---

## Prerequisites

### 1. Install Dependencies

```bash
cd services/orchestration/dashboard
npm install
```

**Note**: If using local Observable services, you may need to:
- Link them: `npm link ../../framework ../../plot ../../runtime ../../stdlib`
- Or install from npm: The package.json already references npm versions

### 2. Set Environment Variables

```bash
# Set Perplexity API base URL
export PERPLEXITY_API_BASE=http://localhost:8080

# Or create .env file
echo "PERPLEXITY_API_BASE=http://localhost:8080" > .env
```

### 3. Start Perplexity API (if not running)

The dashboards need the Perplexity API to be running. Make sure:
- Orchestration service is running on port 8080
- Perplexity API endpoints are accessible
- You have at least one processed request to test with

---

## Testing Steps

### Step 1: Verify Project Structure

```bash
cd services/orchestration/dashboard
tree -L 3
```

Expected structure:
```
dashboard/
├── src/
│   ├── index.md
│   ├── processing.md
│   ├── results.md
│   ├── analytics.md
│   └── styles.css
├── data/
│   └── loaders/
│       ├── processing.js
│       ├── results.js
│       ├── intelligence.js
│       └── analytics.js
├── observable.config.js
└── package.json
```

### Step 2: Check Dependencies

```bash
npm list @observablehq/framework @observablehq/plot @observablehq/runtime @observablehq/stdlib
```

All should be installed. If not:
```bash
npm install
```

### Step 3: Start Development Server

```bash
npm run dev
```

This should:
- Start Observable Framework preview server
- Open browser (or provide URL like `http://localhost:3000`)
- Show the landing page

### Step 4: Test Landing Page

1. Navigate to `http://localhost:3000` (or provided URL)
2. Verify:
   - ✅ Landing page loads
   - ✅ Navigation cards are visible
   - ✅ Links to Processing, Results, Analytics work
   - ✅ Design system styles are applied (colors, typography, spacing)

### Step 5: Test Processing Dashboard

1. Navigate to `/processing`
2. Test without request ID:
   - ✅ Should show "No Request Selected" empty state
   - ✅ Should be inviting, not scary
3. Test with request ID:
   - Add `?request_id=YOUR_REQUEST_ID` to URL
   - Or use input field if available
   - Verify:
     - ✅ Processing status card shows metrics
     - ✅ Progress bar displays correctly
     - ✅ Charts render (may be empty if no data)
     - ✅ Current step displays
     - ✅ Error handling works gracefully

**Test Request IDs**: Use actual request IDs from your Perplexity API, or create test data.

### Step 6: Test Results Dashboard

1. Navigate to `/results`
2. Test without request ID:
   - ✅ Should show empty state
3. Test with request ID:
   - Add `?request_id=YOUR_REQUEST_ID` to URL
   - Verify:
     - ✅ Intelligence summary card shows counts
     - ✅ Domain distribution chart renders
     - ✅ Relationship network diagram displays
     - ✅ Pattern frequency chart shows data
     - ✅ Processing time histogram renders
     - ✅ Documents list displays

### Step 7: Test Analytics Dashboard

1. Navigate to `/analytics`
2. Verify:
   - ✅ Analytics summary card shows metrics
   - ✅ Request volume time series chart renders
   - ✅ Success rate trends chart displays
   - ✅ Domain treemap shows distribution
   - ✅ Performance metrics chart renders
   - ✅ Recent activity list displays

**Note**: Analytics may be empty if no requests exist yet.

---

## Common Issues & Solutions

### Issue 1: "Module not found" errors

**Solution**:
```bash
# Reinstall dependencies
rm -rf node_modules package-lock.json
npm install
```

### Issue 2: Charts not rendering

**Possible causes**:
- No data available (expected for empty states)
- JavaScript errors in browser console
- Observable Plot not loaded

**Solution**:
- Check browser console for errors
- Verify data loaders return correct format
- Ensure Observable Plot is imported correctly

### Issue 3: API connection errors

**Solution**:
- Verify `PERPLEXITY_API_BASE` is set correctly
- Check if Perplexity API is running
- Test API endpoints directly with curl:
  ```bash
  curl http://localhost:8080/api/perplexity/status/YOUR_REQUEST_ID
  ```

### Issue 4: Styling not applied

**Solution**:
- Verify `src/styles.css` exists
- Check if Framework is loading CSS correctly
- Inspect elements to see if styles are applied

---

## Manual Testing Checklist

### Visual Design
- [ ] Colors match design system (iOS blue `#007AFF`, etc.)
- [ ] Typography is readable and consistent
- [ ] Spacing is generous and consistent
- [ ] Cards have proper shadows and hover effects
- [ ] Animations are smooth (fade-in, slide-up)

### Functionality
- [ ] All three dashboards load
- [ ] Navigation works between pages
- [ ] Data loaders fetch from API correctly
- [ ] Charts render with data
- [ ] Empty states are inviting
- [ ] Error states are helpful

### Responsiveness
- [ ] Layout works on desktop (1200px+)
- [ ] Layout adapts to tablet (768px)
- [ ] Layout works on mobile (375px)
- [ ] Charts scale appropriately

### Performance
- [ ] Pages load quickly
- [ ] Charts render smoothly
- [ ] No console errors
- [ ] No memory leaks

---

## Test Data

### Create Test Request

To test with real data, create a processing request:

```bash
curl -X POST http://localhost:8080/api/perplexity/process \
  -H "Content-Type: application/json" \
  -d '{
    "query": "test query",
    "limit": 5,
    "async": false
  }'
```

Save the `request_id` from the response and use it in the dashboards.

### Mock Data (for development)

You can create mock data loaders for testing without API:

```javascript
// data/loaders/processing-mock.js
export default async function(requestId) {
  return {
    request_id: requestId,
    status: "completed",
    progress_percent: 100,
    statistics: {
      documents_processed: 5,
      documents_succeeded: 5,
      documents_failed: 0
    },
    completed_steps: ["connect", "extract", "process", "catalog", "train", "localai", "search"],
    current_step: "completed",
    created_at: new Date().toISOString()
  };
}
```

---

## Browser Testing

Test in multiple browsers:
- [ ] Chrome/Edge (Chromium)
- [ ] Firefox
- [ ] Safari
- [ ] Mobile browsers (iOS Safari, Chrome Mobile)

---

## Next Steps After Testing

Once Phase 2 is verified:
1. ✅ Fix any issues found
2. ✅ Document any limitations
3. ✅ Proceed to Phase 3 (Knowledge Graph, Query Dashboard, Real-time Updates)

---

## Reporting Issues

If you find issues:
1. Check browser console for errors
2. Check server logs
3. Verify API endpoints are working
4. Document the issue with:
   - Steps to reproduce
   - Expected behavior
   - Actual behavior
   - Browser/OS information

---

**Happy Testing!** 🎉


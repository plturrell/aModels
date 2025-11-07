# Phase 2 Testing & Review Guide

## Quick Start Testing

### 1. Run Setup Script

```bash
cd services/orchestration/dashboard
./test-setup.sh
```

This will:
- ✅ Check Node.js version (>= 18)
- ✅ Install dependencies
- ✅ Create .env file
- ✅ Verify project structure
- ✅ Check JavaScript syntax

### 2. Start Development Server

```bash
npm run dev
```

The server will start and provide a URL (usually `http://localhost:3000`).

### 3. Test Each Dashboard

#### Landing Page (`/`)
- [ ] Page loads
- [ ] Navigation cards visible
- [ ] Links work
- [ ] Design system applied (colors, typography, spacing)

#### Processing Dashboard (`/processing`)
- [ ] Empty state shows when no request ID
- [ ] With request ID: status card displays
- [ ] Progress bar works
- [ ] Charts render (may be empty if no data)
- [ ] Current step displays

#### Results Dashboard (`/results`)
- [ ] Empty state shows when no request ID
- [ ] With request ID: intelligence summary displays
- [ ] Domain distribution chart renders
- [ ] Relationship network diagram displays
- [ ] Pattern frequency chart shows
- [ ] Documents list displays

#### Analytics Dashboard (`/analytics`)
- [ ] Analytics summary card shows
- [ ] Request volume chart renders
- [ ] Success rate chart displays
- [ ] Domain treemap shows
- [ ] Performance metrics chart renders
- [ ] Recent activity list displays

---

## Detailed Testing Checklist

### Visual Design Review

#### Colors
- [ ] Primary blue (`#007AFF`) used consistently
- [ ] Success green (`#34C759`) for positive metrics
- [ ] Error red (`#FF3B30`) only when needed
- [ ] Gray scale for text hierarchy
- [ ] Colors guide attention, don't distract

#### Typography
- [ ] System fonts (SF Pro) load correctly
- [ ] Clear hierarchy (32px → 20px → 17px → 14px)
- [ ] Readable line heights
- [ ] Consistent weights (400, 500, 600)

#### Spacing
- [ ] Generous whitespace (24px, 32px, 48px)
- [ ] Consistent gaps
- [ ] Cards have proper padding
- [ ] Charts have proper margins

#### Components
- [ ] Cards have shadows and hover effects
- [ ] Buttons have proper styling
- [ ] Progress bars are clear
- [ ] Empty states are inviting
- [ ] Error states are helpful

### Functionality Review

#### Data Loading
- [ ] Loaders fetch from API correctly
- [ ] Error handling works gracefully
- [ ] Empty states show when no data
- [ ] Loading states are smooth

#### Charts
- [ ] All charts render without errors
- [ ] Charts scale appropriately
- [ ] Colors are purposeful
- [ ] Labels are readable
- [ ] Legends work correctly

#### Navigation
- [ ] Links between pages work
- [ ] URL parameters work (request_id)
- [ ] Back/forward navigation works
- [ ] Deep linking works

### Responsiveness Review

#### Desktop (1200px+)
- [ ] Layout is centered
- [ ] Charts are full width
- [ ] Cards are properly sized
- [ ] Navigation is clear

#### Tablet (768px)
- [ ] Layout adapts
- [ ] Charts scale down
- [ ] Cards stack properly
- [ ] Navigation is accessible

#### Mobile (375px)
- [ ] Layout is mobile-friendly
- [ ] Charts are readable
- [ ] Cards are full width
- [ ] Navigation is usable

### Performance Review

- [ ] Pages load quickly (< 2 seconds)
- [ ] Charts render smoothly
- [ ] No console errors
- [ ] No memory leaks
- [ ] Smooth animations

---

## Common Issues & Fixes

### Issue: "Cannot find module '@observablehq/plot'"

**Fix**:
```bash
npm install @observablehq/plot @observablehq/framework @observablehq/runtime @observablehq/stdlib
```

### Issue: Charts not rendering

**Check**:
1. Browser console for errors
2. Data format from loaders
3. Observable Plot import

**Fix**: Ensure data is in correct format expected by Plot

### Issue: API connection errors

**Check**:
1. `PERPLEXITY_API_BASE` environment variable
2. API is running on correct port
3. CORS is configured

**Fix**: Update `.env` file or set environment variable

### Issue: Styling not applied

**Check**:
1. `src/styles.css` exists
2. Framework is loading CSS
3. Browser DevTools shows styles

**Fix**: Verify CSS is being imported correctly

---

## Test Data

### Create Test Request

```bash
curl -X POST http://localhost:8080/api/perplexity/process \
  -H "Content-Type: application/json" \
  -d '{
    "query": "artificial intelligence research",
    "limit": 5,
    "async": false
  }'
```

Save the `request_id` from response.

### Use Test Request ID

Add to dashboard URLs:
- `/processing?request_id=YOUR_REQUEST_ID`
- `/results?request_id=YOUR_REQUEST_ID`

---

## Browser Testing

Test in:
- [ ] Chrome/Edge (Chromium)
- [ ] Firefox
- [ ] Safari
- [ ] Mobile Safari
- [ ] Chrome Mobile

---

## Performance Benchmarks

Target metrics:
- **Page Load**: < 2 seconds
- **Chart Render**: < 500ms
- **Data Fetch**: < 1 second
- **Animation**: 60fps

---

## Accessibility Testing

- [ ] Keyboard navigation works
- [ ] Screen reader compatible
- [ ] Color contrast meets WCAG AA
- [ ] Focus indicators visible
- [ ] Alt text for images

---

## Next Steps

Once testing is complete:
1. ✅ Document any issues found
2. ✅ Fix critical bugs
3. ✅ Update documentation
4. ✅ Proceed to Phase 3

---

**Happy Testing!** 🎉


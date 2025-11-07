# Browser Shell Integration Test Guide

## Quick Start

### Terminal 1: Start Dashboard
```bash
cd services/orchestration/dashboard
npm run dev
```

**Expected Output**:
- Server starts on `http://localhost:3000`
- Observable Framework preview server running
- Dashboard accessible in browser

### Terminal 2: Start Browser Shell
```bash
cd services/browser/shell
npm start
```

**Expected Output**:
- Vite builds React UI
- Electron app launches
- Browser Shell window opens

---

## Testing the Integration

### Step 1: Verify Dashboard is Running

Open browser and navigate to:
```
http://localhost:3000
```

**Expected**:
- ✅ Landing page loads
- ✅ Navigation cards visible
- ✅ Links to Processing, Results, Analytics work

### Step 2: Verify Browser Shell is Running

**Expected**:
- ✅ Electron window opens
- ✅ Left panel: Chromium browser
- ✅ Right panel: React control panel
- ✅ Navigation sidebar visible

### Step 3: Access Perplexity Module

**Option A: Via Sidebar**
1. Look for "Perplexity" in the navigation sidebar
2. Click on "Perplexity"
3. Description should show: "Processing results & analytics"
4. Icon: Dashboard icon

**Option B: Via Home Page**
1. Click "Home" in sidebar
2. Look for "Perplexity Dashboard" in Quick Links
3. Click the link
4. Should navigate to Perplexity module

### Step 4: Verify Dashboard Loads

**Expected**:
- ✅ Loading spinner appears briefly
- ✅ Dashboard iframe loads
- ✅ Observable Framework dashboard visible
- ✅ Can navigate within dashboard (Processing, Results, Analytics)
- ✅ Charts render correctly

### Step 5: Test Dashboard Navigation

Within the embedded dashboard:
1. Click "Processing" - should show processing dashboard
2. Click "Results" - should show results dashboard
3. Click "Analytics" - should show analytics dashboard
4. All should work within the iframe

---

## Troubleshooting

### Dashboard Not Loading

**Symptoms**:
- Error message: "Failed to load Perplexity Dashboard"
- Blank iframe
- Loading spinner never stops

**Solutions**:
1. Verify dashboard server is running:
   ```bash
   curl http://localhost:3000
   ```

2. Check dashboard URL in Browser Shell:
   - Default: `http://localhost:3000`
   - Can be configured via `VITE_DASHBOARD_URL`

3. Check browser console for errors:
   - Open DevTools in Electron
   - Look for CORS or network errors

4. Verify dashboard is accessible standalone:
   - Open `http://localhost:3000` in regular browser
   - Should work independently

### Browser Shell Not Starting

**Symptoms**:
- Electron window doesn't open
- Build errors
- Module not found errors

**Solutions**:
1. Install dependencies:
   ```bash
   cd services/browser/shell/ui
   npm install
   ```

2. Check TypeScript compilation:
   ```bash
   npm run build
   ```

3. Verify all modules are imported correctly

### Perplexity Module Not Visible

**Symptoms**:
- "Perplexity" not in sidebar
- Navigation item missing

**Solutions**:
1. Verify files were created:
   ```bash
   ls services/browser/shell/ui/src/modules/Perplexity/
   ```

2. Check TypeScript compilation:
   ```bash
   cd services/browser/shell/ui
   npm run build
   ```

3. Restart Browser Shell

---

## Expected Behavior

### Successful Integration

✅ **Dashboard Server**:
- Running on port 3000
- Observable Framework serving pages
- All dashboards accessible

✅ **Browser Shell**:
- Electron window open
- Perplexity in navigation
- Module loads when clicked
- Dashboard visible in iframe

✅ **User Experience**:
- Smooth navigation
- No errors in console
- Charts render correctly
- Can interact with dashboard

---

## Configuration

### Dashboard URL

Set in `services/browser/shell/ui/.env`:
```bash
VITE_DASHBOARD_URL=http://localhost:3000
```

Or when starting:
```bash
VITE_DASHBOARD_URL=http://localhost:3000 npm start
```

### Dashboard API

Set in `services/orchestration/dashboard/.env`:
```bash
PERPLEXITY_API_BASE=http://localhost:8080
```

---

## Next Steps After Testing

1. ✅ Verify both services start correctly
2. ✅ Test navigation to Perplexity module
3. ✅ Verify dashboard loads in iframe
4. ✅ Test dashboard navigation (Processing, Results, Analytics)
5. ✅ Check for any console errors
6. ✅ Verify charts render correctly

---

## Success Criteria

- [ ] Dashboard server starts without errors
- [ ] Browser Shell starts without errors
- [ ] Perplexity appears in navigation
- [ ] Clicking Perplexity loads dashboard
- [ ] Dashboard is fully functional in iframe
- [ ] No console errors
- [ ] Charts render correctly
- [ ] Navigation within dashboard works

---

**Happy Testing!** 🎉


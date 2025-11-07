# Perplexity Module - Runtime Test Guide

## ✅ Integration Status: Ready for Testing

The Perplexity module has been successfully integrated into the Browser Shell as a native React component.

---

## Starting the Browser Shell

### Step 1: Start the Application
```bash
cd services/browser/shell
npm start
```

**Expected Behavior**:
1. Vite builds the React UI (may take 30-60 seconds on first run)
2. Electron window opens automatically
3. Browser Shell interface appears with:
   - Left panel: Chromium browser
   - Right panel: React control panel
   - Sidebar: Navigation with modules

---

## Accessing the Perplexity Module

### Option 1: Via Sidebar Navigation
1. Look for **"Perplexity"** in the left navigation sidebar
2. Icon: 📊 Dashboard icon
3. Description: "Processing results & analytics"
4. Click to open the module

### Option 2: Via Home Page
1. Click **"Home"** in the sidebar
2. Find **"Perplexity Dashboard"** in the Quick Links section
3. Click the link to navigate to the module

---

## Testing Features

### 1. Submit a New Query

**Steps**:
1. Navigate to Perplexity module
2. Find the "New Query" input field at the top
3. Enter a query (e.g., "What is machine learning?")
4. Click "Process" button or press Enter

**Expected Behavior**:
- ✅ Button shows loading spinner
- ✅ Request ID is generated
- ✅ Automatically switches to "Processing" tab
- ✅ Shows processing status

**Verify**:
- Request ID appears in the "Request ID" field
- Status updates in real-time
- Progress bar shows completion percentage

---

### 2. View Processing Status

**Steps**:
1. Enter a request ID in the "Request ID" field, OR
2. Submit a new query (automatically populates request ID)

**Expected Behavior**:
- ✅ **Status Summary Cards**:
  - Documents Processed (total count)
  - Succeeded (green count)
  - Failed (red count if any)

- ✅ **Progress Section**:
  - Current step displayed
  - Progress bar (0-100%)
  - Completed steps list with checkmarks

- ✅ **Request Information**:
  - Request ID (monospace font)
  - Status chip (color-coded: success/error/info)
  - Query text
  - Processing time (if completed)

**Verify**:
- All cards display correct numbers
- Progress bar updates
- Steps show completion status
- Status chip color matches state

---

### 3. View Results and Intelligence

**Steps**:
1. Navigate to "Results" tab
2. Ensure a request ID is set (from previous query)

**Expected Behavior**:
- ✅ **Intelligence Summary Cards**:
  - Domains (count)
  - Relationships (count)
  - Patterns (count)
  - KG Nodes (count)

- ✅ **Processed Documents List**:
  - Document cards with:
    - Title or ID
    - Status chip (success/error)
    - Domain chip (if detected)
    - Relationship count
    - Pattern count
    - Error message (if failed)

- ✅ **Domains Section**:
  - List of detected domains as chips

**Verify**:
- Intelligence cards show correct counts
- Documents list displays all processed documents
- Status chips are color-coded correctly
- Domain chips appear for documents with domains

---

### 4. Browse Analytics

**Steps**:
1. Navigate to "Analytics" tab

**Expected Behavior**:
- ✅ **Summary Cards**:
  - Total Requests
  - Completed (green)
  - Failed (red)
  - Success Rate (percentage)

- ✅ **Performance Metrics**:
  - Average Processing Time (if data available)

- ✅ **Recent Requests Table**:
  - Request ID (truncated)
  - Query text
  - Status chip
  - Documents count
  - Created timestamp

**Verify**:
- Summary cards show aggregate statistics
- Table displays recent requests
- Status chips are color-coded
- Timestamps are formatted correctly

---

## Troubleshooting

### Module Not Visible

**Symptoms**: "Perplexity" doesn't appear in sidebar

**Solutions**:
1. Check if files exist:
   ```bash
   ls services/browser/shell/ui/src/modules/Perplexity/
   ```

2. Rebuild the UI:
   ```bash
   cd services/browser/shell/ui
   npm run build
   ```

3. Restart Browser Shell

---

### API Errors

**Symptoms**: "Failed to fetch" or network errors

**Solutions**:
1. Check API base URL:
   - Default: `http://localhost:8080`
   - Set via `VITE_PERPLEXITY_API_BASE` in `.env`

2. Verify backend is running:
   ```bash
   curl http://localhost:8080/api/perplexity/history
   ```

3. Check CORS settings if accessing from different origin

---

### No Data Displayed

**Symptoms**: Empty views, "No data" messages

**Solutions**:
1. Verify request ID is valid
2. Check if processing completed
3. Ensure backend API is responding:
   ```bash
   curl http://localhost:8080/api/perplexity/status/{request_id}
   ```

---

### TypeScript/Build Errors

**Symptoms**: Build fails, module doesn't load

**Solutions**:
1. Check TypeScript compilation:
   ```bash
   cd services/browser/shell/ui
   npm run build
   ```

2. Verify all dependencies installed:
   ```bash
   npm install
   ```

3. Check for linter errors:
   ```bash
   npm run lint
   ```

---

## Expected UI Behavior

### Loading States
- ✅ Spinner appears while fetching data
- ✅ "Loading..." messages for async operations
- ✅ Disabled buttons during processing

### Error States
- ✅ Red error alerts for failures
- ✅ Helpful error messages
- ✅ Recovery suggestions

### Empty States
- ✅ Informative messages when no data
- ✅ Clear instructions on how to get started
- ✅ Helpful placeholders

### Success States
- ✅ Data displays correctly
- ✅ Cards show accurate counts
- ✅ Status chips reflect current state
- ✅ Progress indicators update

---

## Test Checklist

### Navigation
- [ ] Perplexity appears in sidebar
- [ ] Quick link works on Home page
- [ ] Module loads when clicked
- [ ] Tabs switch correctly

### Query Submission
- [ ] Input field accepts text
- [ ] Process button works
- [ ] Loading state shows
- [ ] Request ID generated
- [ ] Auto-switches to Processing tab

### Processing View
- [ ] Status cards display
- [ ] Progress bar works
- [ ] Steps list shows
- [ ] Request info displays
- [ ] Errors show (if any)

### Results View
- [ ] Intelligence cards display
- [ ] Documents list shows
- [ ] Status chips correct
- [ ] Domain chips appear
- [ ] Counts are accurate

### Analytics View
- [ ] Summary cards display
- [ ] Performance metrics show
- [ ] Recent requests table works
- [ ] Status chips correct
- [ ] Timestamps formatted

### Error Handling
- [ ] Network errors handled
- [ ] API errors displayed
- [ ] Empty states show
- [ ] Loading states work

---

## Success Criteria

✅ **Module Integration**:
- Appears in navigation
- Loads without errors
- Matches other modules' styling

✅ **Functionality**:
- Query submission works
- Status updates correctly
- Results display properly
- Analytics show data

✅ **User Experience**:
- Smooth navigation
- Clear error messages
- Helpful loading states
- Intuitive interface

---

## Next Steps After Testing

1. ✅ Verify all features work
2. ⏭️ Add Observable Plot visualizations (optional)
3. ⏭️ Implement Search tab (optional)
4. ⏭️ Add real-time WebSocket updates (optional)
5. ⏭️ Enhance with relationship graphs (optional)

---

## Status

**Current**: ✅ Ready for Runtime Testing  
**Integration**: ✅ Complete  
**TypeScript**: ✅ Passing  
**Build**: ✅ Successful  

**Ready to test!** Start the Browser Shell and navigate to Perplexity to begin testing! 🎉


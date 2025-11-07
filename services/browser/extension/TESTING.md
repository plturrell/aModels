# Quick Testing Guide - Phase 1 Improvements

## 5-Minute Test Plan

### 1. Load the Extension (1 min)
```bash
# In Chrome/Chromium:
1. Go to chrome://extensions/
2. Enable "Developer mode" (top right toggle)
3. Click "Load unpacked"
4. Select: /Users/user/Documents/aModels/services/browser/extension
```

### 2. First-Run Experience (2 min)

**Welcome wizard should auto-open:**
- ✓ See welcome screen with rocket emoji
- ✓ Click "Get Started"
- ✓ Enter gateway URL (or leave default)
- ✓ Click "Test Connection"
- ✓ See success or error message
- ✓ Click "Start Using aModels"

**If test fails:**
```bash
# Start the gateway first:
cd /Users/user/Documents/aModels
# ... start your gateway service on port 8000
```

### 3. Main Popup (1 min)

**Click extension icon in toolbar:**
- ✓ See connection status (green dot = good!)
- ✓ Try "Extract Text" button
- ✓ Watch loading spinner
- ✓ See success or error message
- ✓ Click "Show Advanced Tools ▼"
- ✓ See more options appear

### 4. Settings Page (1 min)

**Right-click extension → Options:**
- ✓ See current gateway URL
- ✓ Click "Test Connection"
- ✓ See result message
- ✓ Try changing URL and saving

---

## What Should Happen

### ✅ Success Indicators
```
🟢 Connection status: "✓ Connected to gateway"
✅ Success messages: Green background with checkmark
🔄 Loading states: Blue background with spinner
📱 Buttons: Enabled and clickable
```

### ❌ Error Indicators
```
🔴 Connection status: "✗ Gateway offline"
⚠️ Error messages: Red background with recovery steps
🚫 Buttons: Disabled (gray and unclickable)
💡 Help text: Shows what to check
```

---

## Common Issues & Fixes

### Issue: Welcome wizard doesn't appear
**Fix:** Clear extension storage
```javascript
// In popup.js, add temporarily:
chrome.storage.sync.clear();
// Then reload extension
```

### Issue: "Gateway offline" always
**Check:**
1. Is gateway running? `curl http://localhost:8000/healthz`
2. Is URL correct? Check Settings
3. Is CORS enabled on gateway?

### Issue: Buttons don't work
**Fix:**
1. Check browser console (F12)
2. Look for JavaScript errors
3. Verify `errors.js` is loaded

### Issue: Styling looks broken
**Fix:**
1. Hard refresh the popup (Cmd/Ctrl+R)
2. Reload the extension
3. Clear browser cache

---

## Keyboard Shortcuts

```
Tab          Navigate between elements
Enter        Activate focused button
Cmd/Ctrl+Enter   Send chat message
Escape       Close popup
```

---

## Expected User Journey

```
1. Install extension
   ↓
2. Welcome wizard opens automatically
   ↓
3. Configure gateway URL
   ↓
4. Test connection (should succeed)
   ↓
5. See success confirmation
   ↓
6. Click extension icon
   ↓
7. See green connection status
   ↓
8. Try an action (e.g., "Extract Text")
   ↓
9. Watch loading spinner
   ↓
10. See success message
    ↓
✅ DONE! User is onboarded and productive
```

**Time:** <5 minutes from install to first success

---

## Accessibility Testing

### Screen Reader (macOS)
```bash
# Enable VoiceOver
Cmd + F5

# Navigate
VO + Right Arrow    Next item
VO + Left Arrow     Previous item
VO + Space          Activate
```

**Should announce:**
- Button labels clearly
- Connection status
- Loading states
- Error messages

### Keyboard Only
```
# Try completing full workflow without mouse:
Tab → Tab → Enter → Tab → Enter
```

**Should work:**
- Navigate all elements
- Activate all buttons
- Submit forms
- Open links

---

## Performance Check

Open DevTools (F12) → Performance tab:

**Popup load should be:**
- < 50ms initial render
- < 3s connection check
- < 100ms button click response

**Memory usage:**
- Idle: ~5 MB
- Active: ~8 MB
- Peak: <15 MB

---

## Visual Comparison

### Before Phase 1
```
┌─────────────────────┐
│ aModels             │
│ Check health...     │
│ [Check Health]      │
│ ─────────────────── │
│ [Run OCR (demo)]    │
│ [Run SQL (demo)]    │
│ [Telemetry] [Flow]  │
│ [Search] [Redis]    │
│ [Browser]           │
│ ─────────────────── │
│ Prompt: [____]      │
│ Status: Error...    │
└─────────────────────┘
```

### After Phase 1
```
┌──────────────────────────┐
│ aModels                  │
│ 🟢 Connected to gateway  │
│                          │
│ Quick Actions            │
│ 📄 Extract Text          │
│ 🔍 Query Data            │
│ 📊 View Telemetry        │
│ 🌐 Open Browser Shell    │
│                          │
│ ⚡ Show Advanced Tools ▼ │
│                          │
│ LocalAI Chat             │
│ [_______________]        │
│ [Send]                   │
│                          │
│ ✓ Success message        │
│ Settings • Help          │
└──────────────────────────┘
```

---

## Success Checklist

Phase 1 is working if you can:

- [ ] Complete welcome wizard
- [ ] See connection status
- [ ] Click any button and see loading state
- [ ] Get clear error message when gateway is down
- [ ] Open settings and test connection
- [ ] Navigate with keyboard only
- [ ] Read all text with screen reader
- [ ] Understand recovery steps from errors

---

## Report Issues

Found a bug? Document:
1. What you were doing
2. What you expected
3. What actually happened
4. Browser console errors (F12 → Console)
5. Screenshot if visual issue

---

**Happy Testing! 🚀**

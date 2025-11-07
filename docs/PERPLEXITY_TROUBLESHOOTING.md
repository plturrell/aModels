# Perplexity Module - Troubleshooting Guide

## Issue: Perplexity Not Visible in Browser Shell

If you can see the Browser Shell window but cannot see the Perplexity module, follow these steps:

---

## Step 1: Verify Module Files Exist

```bash
cd services/browser/shell/ui
ls -la src/modules/Perplexity/
```

**Expected**: Should see:
- `PerplexityModule.tsx`
- `README.md`
- `views/` directory

---

## Step 2: Rebuild the UI

The UI needs to be rebuilt to include the Perplexity module:

```bash
cd services/browser/shell
npm run build:ui
```

**Expected Output**:
```
✓ built in XXXms
```

---

## Step 3: Restart Browser Shell

After rebuilding, restart the Browser Shell:

```bash
# Kill existing Electron process
pkill -f Electron

# Restart
cd services/browser/shell
npm start
```

---

## Step 4: Check Navigation

Once Browser Shell opens:

1. **Look in the Sidebar**:
   - Scroll through the navigation items
   - Look for "Perplexity" with a Dashboard icon
   - It should be between "Search" and "Home"

2. **Check Home Page**:
   - Click "Home" in the sidebar
   - Look for "Perplexity Dashboard" in Quick Links
   - Click it to navigate

---

## Step 5: Check Browser Console

If still not visible, check for errors:

1. In Electron window, press `Cmd+Option+I` (Mac) or `Ctrl+Shift+I` (Windows/Linux)
2. Open the Console tab
3. Look for any errors related to:
   - `PerplexityModule`
   - `perplexity`
   - Import errors
   - Module not found errors

---

## Common Issues

### Issue 1: Module Not in Build

**Symptom**: Perplexity doesn't appear in navigation

**Solution**:
```bash
cd services/browser/shell
npm run build:ui
```

Then restart Browser Shell.

---

### Issue 2: TypeScript Errors

**Symptom**: Build fails with TypeScript errors

**Solution**:
```bash
cd services/browser/shell/ui
npm run build
```

Fix any TypeScript errors shown, then rebuild.

---

### Issue 3: Module Import Error

**Symptom**: Console shows "Cannot find module" error

**Solution**:
1. Verify file exists: `src/modules/Perplexity/PerplexityModule.tsx`
2. Check import in `App.tsx`:
   ```typescript
   import { PerplexityModule } from "./modules/Perplexity/PerplexityModule";
   ```
3. Rebuild UI

---

### Issue 4: Module Renders But Shows Error

**Symptom**: Perplexity appears but shows error message

**Possible Causes**:
- API backend not running
- Wrong API base URL
- CORS issues

**Solution**:
1. Check API is running: `curl http://localhost:8080/api/perplexity/history`
2. Check `.env` file in `ui/` directory:
   ```bash
   VITE_PERPLEXITY_API_BASE=http://localhost:8080
   ```
3. Rebuild and restart

---

## Verification Checklist

- [ ] Module files exist in `src/modules/Perplexity/`
- [ ] `App.tsx` imports `PerplexityModule`
- [ ] `App.tsx` has `case "perplexity"` in switch
- [ ] `NavPanel.tsx` has Perplexity in `NAV_ITEMS`
- [ ] `useShellStore.ts` includes `"perplexity"` in `ShellModuleId`
- [ ] `HomeModule.tsx` has Perplexity in quickLinks
- [ ] UI build completed successfully
- [ ] Browser Shell restarted after build
- [ ] No console errors

---

## Quick Fix Command

If nothing works, try this complete rebuild:

```bash
cd services/browser/shell

# Clean build
rm -rf ui/dist
rm -rf ui/node_modules/.vite

# Rebuild
npm run build:ui

# Restart
pkill -f Electron
npm start
```

---

## Still Not Working?

1. **Check the build output** for any warnings or errors
2. **Check the Electron console** for runtime errors
3. **Verify all files are saved** and committed
4. **Try a fresh clone** if needed

---

## Expected Behavior

When working correctly:

✅ **Navigation Sidebar**:
- "Perplexity" appears between "Search" and "Home"
- Has Dashboard icon
- Description: "Processing results & analytics"

✅ **Home Page**:
- "Perplexity Dashboard" in Quick Links
- Clicking navigates to Perplexity module

✅ **Perplexity Module**:
- Shows query input field
- Shows request ID field
- Has 4 tabs: Processing, Results, Analytics, Search
- All tabs load without errors

---

**If you've tried all these steps and it still doesn't work, please share:**
1. Build output
2. Console errors (if any)
3. Screenshot of the navigation sidebar


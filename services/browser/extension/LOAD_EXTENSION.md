# How to Load the aModels Extension

## Quick Steps:

1. **Open Chrome**

2. **Go to Extensions page:**
   - Type in address bar: `chrome://extensions/`
   - OR: Menu → Extensions → Manage Extensions

3. **Enable Developer Mode:**
   - Toggle switch in top-right corner
   - Should turn blue when enabled

4. **Click "Load unpacked":**
   - Button appears in top-left after enabling dev mode

5. **Select the extension folder:**
   ```
   /Users/user/Documents/aModels/services/browser/extension
   ```
   - Navigate to this exact folder
   - Click "Select"

6. **Extension loads:**
   - You'll see "aModels" in the list
   - Toggle should be ON (blue)
   - Version: 0.3.0
   - Description: "Press Cmd+K. Do anything."

7. **Pin it (optional):**
   - Click puzzle piece icon in Chrome toolbar
   - Find "aModels"
   - Click pin icon
   - Extension icon appears in toolbar

8. **Test it:**
   - Click the aModels icon
   - Should see purple popup
   - Connection status + Cmd+K hint
   - Press Cmd+K to test

---

## Path to Extension Folder:

```
/Users/user/Documents/aModels/services/browser/extension
```

**Contains:**
- manifest.json
- popup.html
- popup.js
- welcome.html
- options.html
- command-palette.js
- design-system.css
- etc.

---

## What You Should See After Loading:

**In chrome://extensions/:**
```
┌──────────────────────────────────────┐
│ 🔵 aModels                    ON ⬤  │
│ Version 0.3.0                        │
│ Press Cmd+K. Do anything.           │
│ ID: [random ID]                      │
│                                      │
│ Details  Remove  Errors              │
└──────────────────────────────────────┘
```

**When you click the icon:**
```
Small popup (360×240px)
Purple gradient background
🟢 Connected to gateway
⌘K card
```

---

## Troubleshooting:

### "Can't find the folder"
Navigate step by step:
1. Click "Load unpacked"
2. Navigate to /Users
3. Then /user
4. Then /Documents
5. Then /aModels
6. Then /services
7. Then /browser
8. Then /extension ← SELECT THIS
9. Click "Select"

### "Manifest errors"
Check manifest.json exists:
```bash
ls /Users/user/Documents/aModels/services/browser/extension/manifest.json
```

If it exists, the extension should load.

### "Extension loads but looks wrong"
Reload it:
1. Click reload button (🔄) on the extension card
2. Click extension icon again
3. Should show minimal purple popup

---

## Quick Terminal Command:

```bash
# Open Chrome extensions page
open -a "Google Chrome" "chrome://extensions/"

# Then manually do "Load unpacked" and select:
# /Users/user/Documents/aModels/services/browser/extension
```

---

**The extension folder path again (copy this):**
```
/Users/user/Documents/aModels/services/browser/extension
```

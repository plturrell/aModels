# Extension vs Shell - What Goes Where

**IMPORTANT:** The extension and shell are **completely separate** and should look different!

---

## 🔵 Browser Extension (Chrome Extension)

### **What It Is:**
Small popup that appears when you click the extension icon in Chrome

### **What It Should Look Like:**
```
┌─────────────────────────┐
│  Purple gradient        │
│  background             │
│                         │
│  🟢 Connected           │
│                         │
│  ┌──────────────────┐  │
│  │  ⌘K              │  │
│  │  Press Cmd+K     │  │
│  │  Search for any  │  │
│  │  action          │  │
│  └──────────────────┘  │
│                         │
└─────────────────────────┘

Size: 360px × 240px
Background: Purple gradient
Cards: White with glassmorphism
Elements: 3 total (status, hint, command palette)
```

### **Files (Extension Only):**
- `popup.html` - Minimal UI
- `popup.js` - Functions for commands
- `welcome.html` - Auto-connect screen
- `options.html` - Settings page
- `command-palette.js` - Cmd+K interface
- `design-system.css` - Purple gradient styling
- `connection-manager.js` - Auto-reconnect

### **What It Does:**
1. Shows connection status
2. Shows "Press Cmd+K" hint
3. Opens command palette with Cmd+K
4. Commands call popup.js functions
5. Functions make API calls to gateway

### **What It Does NOT Do:**
- ❌ Does NOT have full modules
- ❌ Does NOT have chat interface
- ❌ Does NOT have document grids  
- ❌ Does NOT have SQL editors
- ❌ Does NOT show telemetry dashboards

### **How to Reload:**
```bash
1. Open Chrome
2. Go to chrome://extensions/
3. Find "aModels"  
4. Click reload button (🔄)
5. Click extension icon
6. Should see minimal purple popup
```

---

## 🟢 Electron Shell (Desktop App)

### **What It Is:**
Full desktop application with sidebar and complete modules

### **What It Should Look Like:**
```
┌────────────────────────────────────────────────┐
│ Sidebar       │  Module Content Area           │
│ (Dark/Light)  │  (White background)            │
│               │                                 │
│ • Home        │  [Current Module Display]     │
│ • LocalAI     │  - Home: Quick action cards    │
│ • Documents   │  - LocalAI: Chat messages     │
│ • DMS         │  - Documents: File grid        │
│ • Relational  │  - Telemetry: Live metrics    │
│ • Flows       │  - etc.                        │
│ • Telemetry   │                                 │
│ • Search      │                                 │
│ • Perplexity  │                                 │
│               │                                 │
│ [Theme]       │                                 │
└────────────────────────────────────────────────┘

Window: Full desktop app
Background: Clean white (light) or dark
Cards: Same styling across all modules
Modules: 9 full featured screens
```

### **Files (Shell Only):**
- `/shell/ui/src/modules/Home/HomeModule.tsx`
- `/shell/ui/src/modules/LocalAI/LocalAIModule.tsx`
- `/shell/ui/src/modules/Documents/DocumentsModule.tsx`
- `/shell/ui/src/modules/DMS/DMSModule.tsx`
- `/shell/ui/src/modules/Relational/RelationalModule.tsx`
- `/shell/ui/src/modules/Flows/FlowsModule.tsx`
- `/shell/ui/src/modules/Telemetry/TelemetryModule.tsx`
- `/shell/ui/src/modules/Search/SearchModule.tsx`
- `/shell/ui/src/modules/Perplexity/PerplexityModule.tsx`

### **What It Does:**
- ✅ Full featured modules
- ✅ Chat interface
- ✅ Document management
- ✅ SQL query editor
- ✅ Live telemetry
- ✅ Everything else

### **How to Start:**
```bash
cd /Users/user/Documents/aModels/services/browser/shell
npm start
```

---

## 🚫 Common Confusion

### **"The shell modules are in the extension!"**

**This is WRONG.** They should be completely separate:

**Extension files location:**
```
/services/browser/extension/
  ├── popup.html         (Minimal, purple)
  ├── popup.js           (API calls only)
  ├── command-palette.js (Cmd+K interface)
  └── ...
```

**Shell files location:**
```
/services/browser/shell/ui/src/modules/
  ├── Home/
  ├── LocalAI/
  ├── Documents/
  └── ... (9 modules)
```

**They NEVER mix!**

---

## ✅ How to Check If It's Correct

### **Extension (Browser):**
1. Open Chrome
2. Click extension icon
3. You should see:
   - Purple gradient background
   - Connection status pill
   - "Press Cmd+K" card
   - Nothing else
4. Press Cmd+K
5. Command palette opens
6. Type command, execute
7. Popup might show result briefly

**Size:** Small popup (360×240px)  
**Background:** Purple gradient  
**Content:** Minimal

### **Shell (Electron):**
1. Run `npm start` in `/shell`
2. Desktop app window opens
3. You should see:
   - Sidebar on left
   - Home module with 7 cards
   - Clean white background
   - No purple
4. Click any sidebar item
5. That module loads
6. Full featured interface

**Size:** Full desktop window  
**Background:** White (or dark if toggled)  
**Content:** Complete modules

---

## 🔧 If Something's Wrong

### **Extension shows complex UI:**
**Problem:** Old popup.html still loaded  
**Fix:**
```bash
cd /services/browser/extension
# Check popup.html line count
wc -l popup.html
# Should be ~214 lines

# If it's 391+ lines, it's the old version
# Reload from git or check ALL_MODULES_COMPLETE.md
```

### **Shell modules look different:**
**Problem:** Styling not unified  
**Fix:**
```bash
# We already fixed this!
# index.css is now simple
# styles.css was deleted
# Restart shell: npm start
```

### **Extension has shell modules:**
**Problem:** Manifest loading wrong files  
**Fix:**
```bash
cd extension
cat manifest.json
# Should NOT reference any /shell/ paths
# Should NOT reference React modules
```

---

## 📊 Quick Comparison

| Feature | Extension | Shell |
|---------|-----------|-------|
| **Technology** | Plain HTML/JS | React/TypeScript |
| **Size** | 360×240px popup | Full desktop window |
| **Background** | Purple gradient | White (clean) |
| **UI Elements** | 3 total | 9 modules |
| **Purpose** | Quick commands | Full app |
| **Opens via** | Extension icon | npm start |
| **Navigation** | Command palette | Sidebar + cards |
| **Styling** | design-system.css | index.css + MUI |

---

## 🎯 The Truth

**Extension:** Minimal launcher (Cmd+K)  
**Shell:** Full application (all modules)

**They are NOT the same thing!**  
**They should NOT look the same!**  
**They do NOT share code!**

---

## 🔍 Debug Steps

1. **Close everything**
2. **Reload extension in Chrome**
   - chrome://extensions/
   - Click reload on "aModels"
3. **Check extension popup**
   - Should be purple, minimal
   - Just connection + Cmd+K hint
4. **Kill any running electron**
   - `pkill -f electron`
5. **Start shell fresh**
   - `cd shell && npm start`
6. **Check shell window**
   - Should be white, sidebar, modules
   - No purple

If both look correct, you're done! ✅

---

**Last Updated:** November 7, 2025, 1:20pm

# Perplexity Visibility - Quick Check

## Where to Find Perplexity

### In Navigation Sidebar

The Perplexity module should appear in the **left navigation sidebar**:

1. **Look for these items in order**:
   - LocalAI
   - Documents
   - Flows
   - Telemetry
   - **Search** ← Perplexity should be right after this
   - **Perplexity** ← Should be here with Dashboard icon
   - Home (at the bottom)

2. **If you don't see it**:
   - **Scroll down** in the sidebar - it might be below the visible area
   - Check if the sidebar is collapsed/expanded
   - Look for a Dashboard icon (📊)

### On Home Page

1. Click **"Home"** in the sidebar (bottom item)
2. Look for **"Perplexity Dashboard"** in the Quick Links section
3. It should be the last item in the list
4. Click it to navigate to Perplexity

---

## Quick Test

1. **Open Browser Shell**
2. **Look at the sidebar** - count the navigation items
3. **You should see 7 items**:
   - LocalAI
   - Documents  
   - Flows
   - Telemetry
   - Search
   - **Perplexity** ← This one
   - Home

---

## If Still Not Visible

### Option 1: Check Console
1. In Electron window: `Cmd+Option+I` (Mac) or `Ctrl+Shift+I`
2. Check Console tab for errors
3. Look for any red errors mentioning "Perplexity"

### Option 2: Force Rebuild
```bash
cd services/browser/shell
rm -rf ui/dist
npm run build:ui
pkill -f Electron
npm start
```

### Option 3: Check File
```bash
cd services/browser/shell/ui
cat src/components/NavPanel.tsx | grep -A 5 "perplexity"
```

Should show:
```typescript
{
  id: "perplexity",
  label: "Perplexity",
  description: "Processing results & analytics",
  icon: DashboardIcon
}
```

---

## Visual Guide

**Navigation Sidebar Layout**:
```
┌─────────────────────┐
│   aModels Shell     │
├─────────────────────┤
│ 📱 LocalAI          │
│ 📄 Documents        │
│ 🌳 Flows            │
│ 📊 Telemetry        │
│ 🔍 Search           │
│ 📊 Perplexity  ← HERE│
│ 🏠 Home             │
└─────────────────────┘
```

**Home Page Quick Links**:
```
┌─────────────────────────────┐
│ Quick Links                 │
├─────────────────────────────┤
│ • SGMI Control-M overview   │
│ • Document library          │
│ • Flow orchestrator         │
│ • Telemetry dashboard       │
│ • Semantic search           │
│ • Perplexity Dashboard ← HERE│
└─────────────────────────────┘
```

---

**The module is built and should be visible. If you still can't see it, please check the console for errors!**


# Browser Shell - Web Browser Access Guide

## ✅ Switched to Web Browser Access

The Browser Shell is now accessible through a regular web browser instead of Electron!

---

## Quick Start

### Start the Development Server

```bash
cd services/browser/shell/ui
npm run dev
```

**Server starts on**: `http://localhost:5174`

### Open in Browser

1. Open your web browser (Chrome, Firefox, Safari, etc.)
2. Navigate to: **http://localhost:5174**
3. You should see the Browser Shell interface!

---

## What You'll See

### Navigation Sidebar (Left)
- LocalAI
- Documents
- Flows
- Telemetry
- Search
- **Perplexity** ← Should be visible!
- Home

### Main Content Area (Right)
- Shows the active module
- Full functionality for each module

---

## Accessing Perplexity

### Option 1: Via Sidebar
1. Look for **"Perplexity"** in the left navigation
2. Click it
3. Module loads with tabs: Processing, Results, Analytics, Search

### Option 2: Via Home Page
1. Click **"Home"** in sidebar
2. Find **"Perplexity Dashboard"** in Quick Links
3. Click to navigate

---

## Development vs Production

### Development Mode (Hot Reload)
```bash
npm run dev
```
- Auto-reloads on file changes
- Fast refresh
- Development tools enabled

### Production Preview
```bash
npm run build
npm run serve
```
- Optimized build
- Production-ready
- Served from `dist/` folder

---

## Configuration

### Change Port

Edit `vite.config.ts`:
```typescript
server: {
  port: 5175  // Change to your preferred port
}
```

### API Configuration

Set API base URL in `.env`:
```bash
VITE_PERPLEXITY_API_BASE=http://localhost:8080
VITE_SHELL_API=http://localhost:8000
```

---

## Benefits of Web Browser Access

✅ **Easier Development**: No Electron rebuild needed  
✅ **Faster Iteration**: Hot reload works perfectly  
✅ **Better Debugging**: Browser DevTools available  
✅ **Cross-Platform**: Works on any OS with a browser  
✅ **Easy Sharing**: Just share the URL  
✅ **No Installation**: No Electron dependencies  

---

## Troubleshooting

### Port Already in Use
```bash
# Kill process on port 5174
lsof -ti:5174 | xargs kill -9

# Or change port in vite.config.ts
```

### Module Not Visible
1. Check browser console (F12) for errors
2. Hard refresh: Cmd+Shift+R (Mac) or Ctrl+Shift+R (Windows)
3. Verify `App.tsx` is being used (not `App.jsx`)

### API Calls Failing
- Check API server is running
- Verify CORS settings allow `http://localhost:5174`
- Check API base URL in `.env`

---

## Next Steps

1. ✅ **Start dev server**: `npm run dev`
2. ✅ **Open browser**: `http://localhost:5174`
3. ✅ **Test Perplexity**: Click in sidebar, submit query
4. ✅ **Enjoy**: Full functionality in your browser!

---

**The Browser Shell is now web-accessible!** 🎉


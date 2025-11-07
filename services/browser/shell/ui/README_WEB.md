# Browser Shell - Web Browser Access

## Quick Start

### Option 1: Development Mode (Hot Reload)
```bash
cd services/browser/shell/ui
npm run dev
```

Then open: **http://localhost:5174**

### Option 2: Production Build (Preview)
```bash
cd services/browser/shell/ui
npm run build
npm run serve
```

Then open: **http://localhost:5174**

---

## Accessing the Application

Once the server is running:

1. **Open your web browser**
2. **Navigate to**: `http://localhost:5174`
3. **You should see**:
   - Left sidebar: Navigation with modules
   - Main area: Active module content
   - **Perplexity** should be visible in the sidebar!

---

## Features

✅ **All modules accessible**:
- LocalAI
- Documents
- Flows
- Telemetry
- Search
- **Perplexity** ← Should be visible!
- Home

✅ **Full functionality**:
- Navigation between modules
- Perplexity dashboard with tabs
- Query submission
- Results viewing
- Analytics

---

## Troubleshooting

### Port Already in Use
If port 5174 is taken, change it in `vite.config.ts`:
```typescript
server: {
  port: 5175  // or any other port
}
```

### CORS Issues
If you see CORS errors when calling APIs:
- The API server needs to allow requests from `http://localhost:5174`
- Check API CORS configuration

### Module Not Visible
1. Make sure you're using `App.tsx` (not `App.jsx`)
2. Rebuild: `npm run build`
3. Clear browser cache: Hard refresh (Cmd+Shift+R / Ctrl+Shift+R)

---

## Development Workflow

1. **Start dev server**:
   ```bash
   cd services/browser/shell/ui
   npm run dev
   ```

2. **Open browser**: `http://localhost:5174`

3. **Make changes**: Files auto-reload

4. **Test Perplexity**:
   - Click "Perplexity" in sidebar
   - Submit a query
   - View results

---

## Production Build

For production deployment:

```bash
cd services/browser/shell/ui
npm run build
npm run serve
```

The built files are in `dist/` and can be served by any web server.

---

**Enjoy using the Browser Shell in your web browser!** 🎉


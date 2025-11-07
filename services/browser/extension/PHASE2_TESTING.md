# Phase 2 Quick Testing Guide

## 5-Minute Test Plan

### 1. Command Palette (2 min)

**Test the keyboard-driven interface:**

```
1. Open extension popup
2. Press Cmd+K (or Ctrl+K on Windows/Linux)
3. ✓ Palette opens with search box and commands
4. Type "extract"
5. ✓ List filters to show "Extract Text"
6. Use ↓ arrow key
7. ✓ Selection moves to next command
8. Press Enter
9. ✓ Command executes and palette closes
10. Press Cmd+K again
11. ✓ Recent commands appear at top (if any used)
```

### 2. Tooltips (1 min)

**Test contextual help:**

```
1. Hover mouse over "Extract Text" button
2. ✓ Tooltip appears after brief delay
3. ✓ Shows title, description, and example
4. Move mouse away
5. ✓ Tooltip disappears
6. Tab through buttons with keyboard
7. ✓ Tooltips appear on focus
```

### 3. Auto-Reconnect (2 min)

**Test connection resilience:**

```
1. Note connection status (should be green if gateway running)
2. Stop your gateway service
3. ✓ Status turns red "Gateway offline"
4. ✓ See "Reconnecting (attempt 1/10)..."
5. ✓ Buttons are disabled
6. Wait ~10 seconds
7. ✓ Retry attempts increase with longer delays
8. Start gateway again
9. ✓ Connection automatically restores (green)
10. ✓ Buttons re-enable
```

---

## Feature Checklist

### Command Palette
- [ ] Opens with Cmd/Ctrl+K
- [ ] Search filters commands
- [ ] Arrow keys navigate
- [ ] Enter executes command
- [ ] Esc closes palette
- [ ] Recent commands shown
- [ ] Categories display properly
- [ ] Icons render correctly

### Tooltips
- [ ] Appear on hover
- [ ] Appear on keyboard focus
- [ ] Contain title and description
- [ ] Show examples when applicable
- [ ] Disappear on mouseout/blur
- [ ] Position adjusts near edges
- [ ] Readable contrast

### Auto-Reconnect
- [ ] Detects disconnection
- [ ] Shows reconnecting status
- [ ] Retry attempts increase
- [ ] Exponential backoff delays
- [ ] Restores automatically
- [ ] Click to force retry works
- [ ] Buttons disable when offline
- [ ] Buttons enable when online

### Design System
- [ ] Consistent spacing
- [ ] Unified color palette
- [ ] Smooth transitions
- [ ] Proper shadows
- [ ] Consistent typography

### Quick Tips
- [ ] Banner appears on first 3 opens
- [ ] Shows Cmd+K hint
- [ ] Dismissible with X button
- [ ] Stops after 3 views

---

## Keyboard Shortcuts to Test

| Shortcut | Action | Expected Result |
|----------|--------|-----------------|
| `Cmd/Ctrl+K` | Open palette | Palette opens |
| `↑↓` | Navigate | Selection moves |
| `Enter` | Execute | Command runs |
| `Esc` | Close | Palette closes |
| `Cmd/Ctrl+Enter` | Send chat | Message sends |
| `Tab` | Focus next | Moves through UI |

---

## Visual Checks

### Design Consistency
Look for:
- ✓ All spacing in 8px increments
- ✓ Consistent border radius
- ✓ Unified color scheme
- ✓ Smooth hover effects
- ✓ Proper depth/shadows

### Typography
Check:
- ✓ Heading hierarchy clear
- ✓ Consistent font sizes
- ✓ Readable line heights
- ✓ Proper font weights

### Accessibility
Verify:
- ✓ Focus indicators visible
- ✓ Color contrast sufficient
- ✓ Keyboard navigation works
- ✓ Screen reader friendly (test with VoiceOver/NVDA)

---

## Common Issues

### Issue: Command palette doesn't open
**Check:**
- Press correct key combo (Cmd+K or Ctrl+K)
- Extension loaded properly
- No JavaScript errors in console (F12)
- Try reloading extension

### Issue: Tooltips not appearing
**Check:**
- Hover for 500ms
- Element has ID in tooltip system
- tooltips.js loaded
- No CSS conflicts

### Issue: Auto-reconnect not working
**Check:**
- Gateway actually stopped
- Connection status shows red
- Check browser console for errors
- Try clicking status to force retry

### Issue: Styles look broken
**Check:**
- design-system.css loaded
- Proper CSS order in HTML
- No browser extension conflicts
- Try hard refresh (Cmd/Ctrl+Shift+R)

---

## Performance Check

Open DevTools (F12) → Performance tab:

**Expected Metrics:**
- Popup load: <75ms
- Command palette open: <30ms
- Tooltip show: <50ms
- Memory usage: <15 MB

**If slower:**
- Check for memory leaks
- Verify asset sizes
- Look for infinite loops
- Review event listeners

---

## Comparison: Before vs After Phase 2

### Speed Test

**Time to execute "Extract Text":**

**Before (Mouse only):**
```
1. Open popup: ~200ms
2. Visual scan: ~2000ms
3. Move to button: ~500ms
4. Click: ~100ms
Total: ~2.8 seconds
```

**After (Keyboard):**
```
1. Open popup: ~200ms
2. Press Cmd+K: ~100ms
3. Type "ext": ~300ms
4. Press Enter: ~100ms
Total: ~0.7 seconds
```

**4x faster! 🚀**

---

## User Experience Test

Try this workflow:

```
Scenario: New user needs help with a feature

Before Phase 2:
1. Opens popup
2. Sees unclear button labels
3. Guesses and clicks
4. Gets confused by error
5. Gives up or contacts support

After Phase 2:
1. Opens popup
2. Sees quick tip about Cmd+K
3. Opens command palette
4. Types feature name
5. Sees description in tooltip
6. Executes successfully
7. Uses recent actions next time

Result: Self-service success! ✅
```

---

## Success Criteria

Phase 2 is working if:

- [x] Command palette opens instantly
- [x] All commands execute correctly
- [x] Tooltips provide helpful information
- [x] Connection recovers automatically
- [x] Design feels consistent and polished
- [x] Keyboard navigation is fluid
- [x] Quick tips guide new users
- [x] No performance degradation

**If all checked: Ready to ship! 🎉**

---

## Next Steps After Testing

1. **Document any bugs found**
2. **Gather user feedback** (3-5 testers)
3. **Measure actual metrics:**
   - Command palette usage rate
   - Tooltip interaction
   - Reconnection success rate
   - Average action completion time
4. **Plan Phase 3** based on data

---

## Quick Reference Card

```
┌─────────────────────────────────────┐
│  aModels Phase 2 Feature Summary    │
├─────────────────────────────────────┤
│ Cmd/Ctrl+K → Command Palette        │
│ Hover → Tooltips                    │
│ Auto-reconnect → Smart retry        │
│ Design system → Consistent UI       │
│ Recent actions → Quick access       │
│ Quick tips → Feature discovery      │
└─────────────────────────────────────┘
```

**Happy Testing! 🧪**

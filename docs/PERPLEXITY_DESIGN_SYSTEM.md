# Perplexity Dashboard - Design System
## The Jobs & Ive Lens

> *"Design is not just what it looks like and feels like. Design is how it works."* - Steve Jobs

---

## Design Philosophy

### Core Principles

1. **Simplicity is the Ultimate Sophistication**
   - Remove everything unnecessary
   - One thing at a time
   - Clear purpose for every element

2. **Beauty in Function**
   - Form and function in perfect harmony
   - Every element serves a purpose
   - Visual hierarchy guides naturally

3. **Intuitive by Design**
   - Zero learning curve
   - Obvious interactions
   - Discoverable features

4. **Details Make the Design**
   - Every pixel matters
   - Smooth animations
   - Thoughtful micro-interactions

5. **Human-Centered**
   - Designed for humans, not machines
   - Emotional connection
   - Moments of delight

---

## Color Palette

### Primary Colors
```css
/* iOS-inspired, purposeful colors */
--blue-primary: #007AFF;      /* Primary actions, links */
--blue-light: #5AC8FA;        /* Hover states, highlights */
--blue-dark: #0051D5;         /* Active states, emphasis */

--gray-900: #1d1d1f;          /* Primary text */
--gray-700: #424245;           /* Secondary text */
--gray-500: #86868b;           /* Tertiary text, labels */
--gray-300: #d2d2d7;           /* Borders, dividers */
--gray-100: #f5f5f7;           /* Backgrounds, subtle fills */

--white: #ffffff;              /* Cards, surfaces */
--black: #000000;              /* High contrast text */
```

### Semantic Colors
```css
/* Success, Error, Warning - used sparingly, with purpose */
--green-success: #34C759;      /* Success states, positive metrics */
--red-error: #FF3B30;          /* Errors, critical alerts */
--orange-warning: #FF9500;     /* Warnings, attention needed */
--yellow-info: #FFCC00;         /* Info, neutral alerts */
```

### Chart Colors
```css
/* Purposeful color schemes for visualizations */
--chart-blue: #007AFF;
--chart-green: #34C759;
--chart-orange: #FF9500;
--chart-red: #FF3B30;
--chart-purple: #AF52DE;
--chart-pink: #FF2D55;
--chart-teal: #5AC8FA;
--chart-yellow: #FFCC00;
```

**Usage Guidelines**:
- Use primary blue for interactive elements
- Use grays for text hierarchy (900 → 700 → 500)
- Use semantic colors sparingly, only when needed
- Chart colors should be distinct but harmonious

---

## Typography

### Font Stack
```css
/* System fonts for native feel */
font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", 
             "SF Pro Text", "Helvetica Neue", Arial, sans-serif;
```

### Type Scale
```css
/* Clear hierarchy, generous sizing */
--font-size-display: 48px;     /* Hero text, large numbers */
--font-size-h1: 32px;           /* Page titles */
--font-size-h2: 24px;           /* Section titles */
--font-size-h3: 20px;           /* Card titles */
--font-size-body: 17px;         /* Body text (iOS standard) */
--font-size-small: 14px;        /* Labels, captions */
--font-size-tiny: 12px;         /* Fine print, metadata */

/* Font weights */
--font-weight-light: 300;
--font-weight-regular: 400;
--font-weight-medium: 500;
--font-weight-semibold: 600;
--font-weight-bold: 700;
```

### Typography Guidelines
- **Headings**: Use semibold (600) for hierarchy
- **Body text**: Use regular (400) for readability
- **Labels**: Use medium (500) for emphasis
- **Line height**: 1.5 for body, 1.2 for headings
- **Letter spacing**: -0.01em for headings (tighter)

---

## Spacing System

### Spacing Scale
```css
/* 4px base unit for consistency */
--space-1: 4px;    /* Tight spacing */
--space-2: 8px;    /* Compact spacing */
--space-3: 12px;   /* Default spacing */
--space-4: 16px;   /* Comfortable spacing */
--space-5: 20px;   /* Generous spacing */
--space-6: 24px;   /* Section spacing */
--space-8: 32px;   /* Large section spacing */
--space-10: 40px;  /* Page-level spacing */
--space-12: 48px;  /* Hero spacing */
--space-16: 64px;  /* Maximum spacing */
```

### Usage Guidelines
- **Cards**: 24px padding (generous breathing room)
- **Sections**: 32px vertical spacing
- **Elements**: 16px between related items
- **Groups**: 24px between groups
- **Pages**: 40px top/bottom margins

---

## Components

### Cards
```css
.card {
  background: white;
  border-radius: 12px;
  padding: 24px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.08);
  margin: 16px 0;
}

/* Hover state - subtle, delightful */
.card:hover {
  box-shadow: 0 4px 16px rgba(0,0,0,0.12);
  transform: translateY(-2px);
  transition: all 0.2s ease;
}
```

### Buttons
```css
.button-primary {
  background: #007AFF;
  color: white;
  border: none;
  border-radius: 8px;
  padding: 12px 24px;
  font-size: 17px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;
}

.button-primary:hover {
  background: #0051D5;
  transform: scale(1.02);
}

.button-primary:active {
  transform: scale(0.98);
}
```

### Inputs
```css
.input {
  border: 1px solid #d2d2d7;
  border-radius: 8px;
  padding: 12px 16px;
  font-size: 17px;
  transition: all 0.2s ease;
}

.input:focus {
  outline: none;
  border-color: #007AFF;
  box-shadow: 0 0 0 3px rgba(0,122,255,0.1);
}
```

---

## Animations

### Principles
- **Smooth**: 60fps, no jank
- **Purposeful**: Every animation has a reason
- **Natural**: Ease-in-out curves feel organic
- **Fast**: Users don't wait for animations

### Timing
```css
--duration-fast: 150ms;      /* Micro-interactions */
--duration-normal: 250ms;    /* Standard transitions */
--duration-slow: 400ms;      /* Complex animations */
```

### Easing
```css
--ease-in-out: cubic-bezier(0.4, 0, 0.2, 1);    /* Standard */
--ease-out: cubic-bezier(0, 0, 0.2, 1);         /* Entrances */
--ease-in: cubic-bezier(0.4, 0, 1, 1);          /* Exits */
```

### Common Animations
```css
/* Fade in */
@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}

/* Slide up */
@keyframes slideUp {
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

/* Scale in */
@keyframes scaleIn {
  from {
    opacity: 0;
    transform: scale(0.95);
  }
  to {
    opacity: 1;
    transform: scale(1);
  }
}
```

---

## Layout

### Grid System
```css
/* 12-column grid with generous gutters */
.container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 24px;
}

.grid {
  display: grid;
  grid-template-columns: repeat(12, 1fr);
  gap: 24px;
}
```

### Breakpoints
```css
/* Mobile-first approach */
@media (min-width: 768px) { /* Tablet */ }
@media (min-width: 1024px) { /* Desktop */ }
@media (min-width: 1440px) { /* Large Desktop */ }
```

---

## States

### Loading States
```css
/* Skeleton screens, not spinners */
.skeleton {
  background: linear-gradient(
    90deg,
    #f5f5f7 0%,
    #e5e5ea 50%,
    #f5f5f7 100%
  );
  background-size: 200% 100%;
  animation: shimmer 1.5s infinite;
}

@keyframes shimmer {
  0% { background-position: 200% 0; }
  100% { background-position: -200% 0; }
}
```

### Empty States
```css
/* Inviting, not empty */
.empty-state {
  text-align: center;
  padding: 64px 24px;
  color: #86868b;
}

.empty-state-icon {
  font-size: 64px;
  margin-bottom: 16px;
  opacity: 0.5;
}

.empty-state-title {
  font-size: 20px;
  font-weight: 600;
  color: #1d1d1f;
  margin-bottom: 8px;
}
```

### Error States
```css
/* Helpful, not scary */
.error-state {
  background: #fff5f5;
  border: 1px solid #ffebee;
  border-radius: 8px;
  padding: 16px;
  color: #c62828;
}

.error-state-title {
  font-weight: 600;
  margin-bottom: 8px;
}

.error-state-message {
  font-size: 14px;
  color: #86868b;
}
```

---

## Accessibility

### Guidelines
- **Color contrast**: Minimum 4.5:1 for text
- **Focus states**: Clear, visible focus indicators
- **Keyboard navigation**: All interactions keyboard accessible
- **Screen readers**: Semantic HTML, ARIA labels
- **Touch targets**: Minimum 44x44px for mobile

---

## Design Checklist

### Every Component Should:
- ✅ Have a clear purpose
- ✅ Use consistent spacing
- ✅ Follow typography scale
- ✅ Use purposeful colors
- ✅ Have smooth animations
- ✅ Work on all screen sizes
- ✅ Be accessible
- ✅ Feel delightful

### Every Page Should:
- ✅ Have clear visual hierarchy
- ✅ Use generous whitespace
- ✅ Guide user attention naturally
- ✅ Feel fast and responsive
- ✅ Work without explanation
- ✅ Create moments of delight

---

## Resources

- **Apple Human Interface Guidelines**: https://developer.apple.com/design/human-interface-guidelines/
- **Material Design**: https://material.io/design (for reference)
- **Observable Plot Gallery**: https://observablehq.com/@observablehq/plot-gallery

---

**Remember**: *"Details are not details. They make the design."* - Jony Ive

Every pixel, every animation, every interaction should feel intentional, beautiful, and delightful.


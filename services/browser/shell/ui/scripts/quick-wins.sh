#!/bin/bash
# Quick Wins Script - Get to 9.0/10 in 3 days
# Run this to install dependencies and setup tooling for improvements

set -e

echo "🚀 aModels Shell - Quick Wins Setup"
echo "===================================="
echo ""

# Check if we're in the right directory
if [ ! -f "package.json" ]; then
  echo "❌ Error: Must run from ui/ directory"
  exit 1
fi

echo "📦 Installing dependencies..."
echo ""

# Day 1: Error tracking & notifications
echo "Day 1: Error Tracking & Notifications"
npm install @sentry/react @sentry/vite-plugin
npm install notistack
echo "✅ Sentry and toast notifications ready"
echo ""

# Day 2: Performance & UX
echo "Day 2: Performance & UX"  
npm install react-window web-vitals
echo "✅ Virtual scrolling and Web Vitals ready"
echo ""

# Day 3: Testing & Quality
echo "Day 3: Testing & Quality"
npm install --save-dev @playwright/test @axe-core/react
npx playwright install chromium
echo "✅ Playwright and accessibility testing ready"
echo ""

# Development tools
echo "🛠️  Installing development tools..."
npm install --save-dev vite-bundle-visualizer @lhci/cli
echo "✅ Bundle analyzer and Lighthouse CI ready"
echo ""

# Create directories for new code
echo "📁 Creating directories..."
mkdir -p src/monitoring
mkdir -p src/components/loading-states
mkdir -p src/components/empty-states
mkdir -p e2e
mkdir -p docs
echo "✅ Directories created"
echo ""

echo "✨ Quick wins setup complete!"
echo ""
echo "Next steps:"
echo "1. Configure Sentry (add VITE_SENTRY_DSN to .env)"
echo "2. Implement toast notifications (see ROADMAP_TO_10.md)"
echo "3. Add loading skeletons (templates in components/loading-states/)"
echo "4. Run 'npm run lighthouse' to check current score"
echo ""
echo "See ROADMAP_TO_10.md for detailed implementation guide"

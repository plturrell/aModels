#!/bin/bash
# Phase 2 Testing Setup Script

set -e

echo "🚀 Setting up Phase 2 Dashboard Testing..."
echo ""

# Check Node.js
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js >= 18"
    exit 1
fi

NODE_VERSION=$(node --version | cut -d'v' -f2 | cut -d'.' -f1)
if [ "$NODE_VERSION" -lt 18 ]; then
    echo "❌ Node.js version must be >= 18. Current: $(node --version)"
    exit 1
fi

echo "✅ Node.js $(node --version) detected"

# Check npm
if ! command -v npm &> /dev/null; then
    echo "❌ npm is not installed"
    exit 1
fi

echo "✅ npm $(npm --version) detected"
echo ""

# Navigate to dashboard directory
cd "$(dirname "$0")"

# Check if package.json exists
if [ ! -f "package.json" ]; then
    echo "❌ package.json not found. Are you in the dashboard directory?"
    exit 1
fi

echo "📦 Installing dependencies..."
npm install

echo ""
echo "🔧 Setting up environment..."

# Create .env if it doesn't exist
if [ ! -f ".env" ]; then
    echo "PERPLEXITY_API_BASE=http://localhost:8080" > .env
    echo "✅ Created .env file with default API base URL"
else
    echo "✅ .env file already exists"
fi

echo ""
echo "📋 Verifying project structure..."

# Check required files
REQUIRED_FILES=(
    "observable.config.js"
    "src/index.md"
    "src/processing.md"
    "src/results.md"
    "src/analytics.md"
    "src/styles.css"
    "data/loaders/processing.js"
    "data/loaders/results.js"
    "data/loaders/intelligence.js"
    "data/loaders/analytics.js"
)

MISSING_FILES=()
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        MISSING_FILES+=("$file")
    fi
done

if [ ${#MISSING_FILES[@]} -gt 0 ]; then
    echo "❌ Missing required files:"
    for file in "${MISSING_FILES[@]}"; do
        echo "   - $file"
    done
    exit 1
fi

echo "✅ All required files present"
echo ""

# Check if Observable Framework is available
if npm list @observablehq/framework &> /dev/null; then
    echo "✅ Observable Framework installed"
else
    echo "⚠️  Observable Framework not found. Installing..."
    npm install @observablehq/framework
fi

# Check if Observable Plot is available
if npm list @observablehq/plot &> /dev/null; then
    echo "✅ Observable Plot installed"
else
    echo "⚠️  Observable Plot not found. Installing..."
    npm install @observablehq/plot
fi

# Check if Observable Runtime is available
if npm list @observablehq/runtime &> /dev/null; then
    echo "✅ Observable Runtime installed"
else
    echo "⚠️  Observable Runtime not found. Installing..."
    npm install @observablehq/runtime
fi

# Check if Observable Stdlib is available
if npm list @observablehq/stdlib &> /dev/null; then
    echo "✅ Observable Stdlib installed"
else
    echo "⚠️  Observable Stdlib not found. Installing..."
    npm install @observablehq/stdlib
fi

echo ""
echo "🧪 Running basic syntax checks..."

# Check JavaScript syntax in loaders
for loader in data/loaders/*.js; do
    if node --check "$loader" &> /dev/null; then
        echo "✅ $(basename $loader) syntax OK"
    else
        echo "❌ $(basename $loader) has syntax errors"
        node --check "$loader"
        exit 1
    fi
done

echo ""
echo "✅ Setup complete!"
echo ""
echo "📝 Next steps:"
echo "   1. Start the development server: npm run dev"
echo "   2. Open the URL shown (usually http://localhost:3000)"
echo "   3. Test each dashboard:"
echo "      - Landing page: /"
echo "      - Processing: /processing?request_id=YOUR_ID"
echo "      - Results: /results?request_id=YOUR_ID"
echo "      - Analytics: /analytics"
echo ""
echo "💡 Tip: Make sure the Perplexity API is running on port 8080"
echo "   or update PERPLEXITY_API_BASE in .env"
echo ""


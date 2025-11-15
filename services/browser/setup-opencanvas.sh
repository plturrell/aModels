#!/bin/bash

# Setup script for Open Canvas integration with aModels
# This script helps configure and start Open Canvas with LocalAI

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENCANVAS_DIR="$SCRIPT_DIR/open-canvas"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Open Canvas is cloned
check_opencanvas() {
    if [ ! -d "$OPENCANVAS_DIR" ]; then
        error "Open Canvas not found at $OPENCANVAS_DIR"
        info "Cloning Open Canvas repository..."
        git clone https://github.com/langchain-ai/open-canvas.git "$OPENCANVAS_DIR"
        success "Open Canvas cloned successfully"
    else
        success "Open Canvas found at $OPENCANVAS_DIR"
    fi
}

# Check dependencies
check_dependencies() {
    info "Checking dependencies..."
    
    # Check Node.js
    if ! command -v node &> /dev/null; then
        error "Node.js is not installed. Please install Node.js 18 or higher."
        exit 1
    fi
    success "Node.js found: $(node --version)"
    
    # Check Yarn
    if ! command -v yarn &> /dev/null; then
        error "Yarn is not installed. Installing yarn..."
        npm install -g yarn
    fi
    success "Yarn found: $(yarn --version)"
    
    # Check Python (for LangGraph)
    if ! command -v python3 &> /dev/null; then
        error "Python 3 is not installed. Please install Python 3.8 or higher."
        exit 1
    fi
    success "Python found: $(python3 --version)"
    
    # Check pip
    if ! command -v pip3 &> /dev/null; then
        error "pip3 is not installed. Please install pip."
        exit 1
    fi
    success "pip3 found"
}

# Install Node dependencies
install_node_deps() {
    info "Installing Node.js dependencies..."
    cd "$OPENCANVAS_DIR"
    yarn install
    success "Node.js dependencies installed"
}

# Install Python dependencies
install_python_deps() {
    info "Installing Python dependencies..."
    
    # Check if langgraph-cli is installed
    if ! command -v langgraph &> /dev/null; then
        info "Installing LangGraph CLI..."
        pip3 install langgraph-cli
        success "LangGraph CLI installed"
    else
        success "LangGraph CLI already installed"
    fi
}

# Configure environment variables
configure_env() {
    info "Configuring environment variables..."
    
    # Root .env
    if [ ! -f "$OPENCANVAS_DIR/.env" ]; then
        cp "$OPENCANVAS_DIR/.env.example" "$OPENCANVAS_DIR/.env"
        
        # Configure for LocalAI
        cat > "$OPENCANVAS_DIR/.env" << EOF
# LangSmith tracing (optional)
LANGSMITH_TRACING="false"
LANGSMITH_API_KEY=""

# LocalAI integration
OPENAI_API_KEY="sk-local-key-not-required"
OPENAI_API_BASE="http://localhost:8080/v1"

# Anthropic (optional fallback)
ANTHROPIC_API_KEY=""

# Fireworks (optional)
FIREWORKS_API_KEY=""

# Gemini (optional)
GOOGLE_API_KEY=""

# Groq - STT (optional)
GROQ_API_KEY=""

# Supabase (update these with your values)
NEXT_PUBLIC_SUPABASE_URL="https://your-project.supabase.co"
NEXT_PUBLIC_SUPABASE_ANON_KEY="your-anon-key-here"
SUPABASE_SERVICE_ROLE="your-service-role-key-here"

# FireCrawl (optional)
FIRECRAWL_API_KEY=""
EOF
        success "Root .env file created"
        warning "Please update Supabase credentials in $OPENCANVAS_DIR/.env"
    else
        info "Root .env file already exists"
    fi
    
    # Web app .env
    if [ ! -f "$OPENCANVAS_DIR/apps/web/.env" ]; then
        cp "$OPENCANVAS_DIR/apps/web/.env.example" "$OPENCANVAS_DIR/apps/web/.env"
        
        cat > "$OPENCANVAS_DIR/apps/web/.env" << EOF
# Feature flags for hiding/showing specific models
NEXT_PUBLIC_FIREWORKS_ENABLED=false
NEXT_PUBLIC_GEMINI_ENABLED=false
NEXT_PUBLIC_ANTHROPIC_ENABLED=false
NEXT_PUBLIC_OPENAI_ENABLED=true
NEXT_PUBLIC_AZURE_ENABLED=false
NEXT_PUBLIC_OLLAMA_ENABLED=false
NEXT_PUBLIC_GROQ_ENABLED=false

# Supabase for authentication (update these with your values)
NEXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=your-anon-key-here
NEXT_PUBLIC_SUPABASE_URL_DOCUMENTS=https://your-project.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY_DOCUMENTS=your-anon-key-here

# For transcription (optional)
GROQ_API_KEY=

# For web scraping (optional)
FIRECRAWL_API_KEY=
EOF
        success "Web app .env file created"
        warning "Please update Supabase credentials in $OPENCANVAS_DIR/apps/web/.env"
    else
        info "Web app .env file already exists"
    fi
}

# Check LocalAI status
check_localai() {
    info "Checking LocalAI status..."
    
    if curl -s http://localhost:8080/v1/models &> /dev/null; then
        success "LocalAI is running on port 8080"
        info "Available models:"
        curl -s http://localhost:8080/v1/models | python3 -m json.tool | grep '"id"' || echo "  Could not retrieve models"
    else
        warning "LocalAI is not running on port 8080"
        warning "Please start LocalAI before using Open Canvas"
        warning "Run: make -f Makefile.services start-localai"
    fi
}

# Create startup scripts
create_startup_scripts() {
    info "Creating startup scripts..."
    
    # Start LangGraph script
    cat > "$OPENCANVAS_DIR/start-langgraph.sh" << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
echo "Starting LangGraph server..."
langgraph dev --host 0.0.0.0 --port 2024
EOF
    chmod +x "$OPENCANVAS_DIR/start-langgraph.sh"
    success "Created start-langgraph.sh"
    
    # Start web app script
    cat > "$OPENCANVAS_DIR/start-web.sh" << 'EOF'
#!/bin/bash
cd "$(dirname "$0")/apps/web"
echo "Starting Open Canvas web application..."
yarn dev
EOF
    chmod +x "$OPENCANVAS_DIR/start-web.sh"
    success "Created start-web.sh"
    
    # Combined startup script
    cat > "$OPENCANVAS_DIR/start-all.sh" << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"

echo "Starting Open Canvas services..."

# Start LangGraph in background
echo "Starting LangGraph server on port 2024..."
./start-langgraph.sh &
LANGGRAPH_PID=$!

# Wait for LangGraph to be ready
sleep 5

# Start web app
echo "Starting web application on port 3000..."
./start-web.sh &
WEB_PID=$!

echo ""
echo "==================================="
echo "Open Canvas is starting..."
echo "==================================="
echo "LangGraph Server: http://localhost:2024"
echo "Web Application: http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop all services"
echo "==================================="

# Wait for Ctrl+C
trap "kill $LANGGRAPH_PID $WEB_PID 2>/dev/null; exit" INT
wait
EOF
    chmod +x "$OPENCANVAS_DIR/start-all.sh"
    success "Created start-all.sh"
}

# Print instructions
print_instructions() {
    echo ""
    echo "========================================="
    echo "  Open Canvas Setup Complete!"
    echo "========================================="
    echo ""
    echo "Next steps:"
    echo ""
    echo "1. Configure Supabase:"
    echo "   - Sign up at https://supabase.com"
    echo "   - Create a new project"
    echo "   - Update credentials in:"
    echo "     - $OPENCANVAS_DIR/.env"
    echo "     - $OPENCANVAS_DIR/apps/web/.env"
    echo ""
    echo "2. Ensure LocalAI is running:"
    echo "   make -f Makefile.services start-localai"
    echo ""
    echo "3. Start Open Canvas:"
    echo "   cd $OPENCANVAS_DIR"
    echo "   ./start-all.sh"
    echo ""
    echo "   Or start services separately:"
    echo "   Terminal 1: ./start-langgraph.sh"
    echo "   Terminal 2: ./start-web.sh"
    echo ""
    echo "4. Access the applications:"
    echo "   - Open Canvas: http://localhost:3000"
    echo "   - Shell UI: http://localhost:4173"
    echo "   - LangGraph: http://localhost:2024"
    echo ""
    echo "========================================="
    echo ""
    echo "For more information, see:"
    echo "  $SCRIPT_DIR/OPEN_CANVAS_INTEGRATION.md"
    echo ""
}

# Main setup flow
main() {
    echo ""
    echo "========================================="
    echo "  Open Canvas Integration Setup"
    echo "========================================="
    echo ""
    
    check_opencanvas
    check_dependencies
    install_node_deps
    install_python_deps
    configure_env
    create_startup_scripts
    check_localai
    print_instructions
}

# Run main function
main

#!/bin/bash
# ╔══════════════════════════════════════════════════════════════╗
# ║  PRAJNA — Complete Setup Script                              ║
# ║  Run this on any Mac (M1/M2/M4) to set up the full platform ║
# ╚══════════════════════════════════════════════════════════════╝

set -e
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log()  { echo -e "${GREEN}[✓]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }
err()  { echo -e "${RED}[✗]${NC} $1"; }
info() { echo -e "${BLUE}[i]${NC} $1"; }

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  🧠 PRAJNA Setup                                        ║"
echo "║  Predictive Resource Allocation for JEE/NEET Aspirants   ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Components:                                             ║"
echo "║  • Prediction Engine (Python FastAPI)                    ║"
echo "║  • Student/Org Backend (Node.js Express)                 ║"
echo "║  • Unified Portal (Next.js 16)                           ║"
echo "║  • AI Models: Llama 3.2 + Qwen2.5-VL + Aryabhata 1.0   ║"
echo "║  • Question Bank: 1.14M questions (qbg.db)              ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
PORTAL_DIR="$(dirname "$BASE_DIR")/prajna-portal"

# ── Step 1: Check prerequisites ──────────────────────────────
info "Step 1: Checking prerequisites..."

# Python
if command -v python3 &>/dev/null; then
    log "Python3 $(python3 --version 2>&1 | awk '{print $2}')"
else
    err "Python3 not found. Install from https://python.org"
    exit 1
fi

# Node.js
if command -v node &>/dev/null; then
    log "Node.js $(node --version)"
else
    err "Node.js not found. Install from https://nodejs.org"
    exit 1
fi

# npm
if command -v npm &>/dev/null; then
    log "npm $(npm --version)"
else
    err "npm not found"
    exit 1
fi

# pm2
if command -v pm2 &>/dev/null; then
    log "pm2 installed"
else
    warn "pm2 not found. Installing globally..."
    npm install -g pm2
    log "pm2 installed"
fi

# Check Apple Silicon
CHIP=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Unknown")
RAM=$(sysctl -n hw.memsize 2>/dev/null | awk '{printf "%.0f", $0/1073741824}')
log "Hardware: $CHIP, ${RAM}GB RAM"

if [[ "$RAM" -lt 16 ]]; then
    warn "16GB+ RAM recommended. You have ${RAM}GB. Models may run slowly."
fi

# ── Step 2: Python virtual environment ───────────────────────
info "Step 2: Setting up Python environment..."

cd "$BASE_DIR"

if [ ! -d "venv" ]; then
    python3 -m venv venv
    log "Created virtual environment"
else
    log "Virtual environment exists"
fi

source venv/bin/activate

pip install --quiet -r requirements.txt
pip install --quiet httpx PyMuPDF mlx mlx-lm huggingface_hub google-generativeai
log "Python dependencies installed"

# ── Step 3: Node.js backend ──────────────────────────────────
info "Step 3: Setting up Node.js backend..."

cd "$BASE_DIR/backend"
if [ ! -d "node_modules" ]; then
    npm install --silent
    log "Backend dependencies installed"
else
    log "Backend dependencies exist"
fi

# Create .env if not exists
if [ ! -f ".env" ]; then
    cat > .env << 'ENVEOF'
JWT_SECRET=prajna-secret-key-change-in-production
PORT=4000
FRONTEND_ORIGIN=*
ENVEOF
    log "Created backend .env"
else
    log "Backend .env exists"
fi

# ── Step 4: Ollama + AI Models ───────────────────────────────
info "Step 4: Setting up Ollama and AI models..."

# Install Ollama if not present
if command -v ollama &>/dev/null || [ -f "/Applications/Ollama.app/Contents/MacOS/Ollama" ]; then
    log "Ollama installed"
else
    warn "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh 2>/dev/null || {
        warn "Ollama install needs sudo. Run manually: curl -fsSL https://ollama.com/install.sh | sh"
    }
fi

# Start Ollama server if not running
if curl -s http://localhost:11434/api/tags &>/dev/null; then
    log "Ollama server running"
else
    warn "Starting Ollama server..."
    if [ -f "/Applications/Ollama.app/Contents/MacOS/Ollama" ]; then
        /Applications/Ollama.app/Contents/MacOS/Ollama serve &>/tmp/ollama.log &
    else
        ollama serve &>/tmp/ollama.log &
    fi
    sleep 5
    if curl -s http://localhost:11434/api/tags &>/dev/null; then
        log "Ollama server started"
    else
        err "Could not start Ollama. Start it manually: ollama serve"
    fi
fi

# Pull models
info "Pulling AI models (this may take 10-20 minutes on first run)..."

pull_model() {
    local model=$1
    local desc=$2
    if curl -s http://localhost:11434/api/tags 2>/dev/null | python3 -c "import sys,json; models=[m['name'] for m in json.load(sys.stdin).get('models',[])]; sys.exit(0 if '$model' in models else 1)" 2>/dev/null; then
        log "$desc already downloaded"
    else
        info "Downloading $desc..."
        curl -s http://localhost:11434/api/pull -d "{\"name\":\"$model\"}" | tail -1
        log "$desc downloaded"
    fi
}

pull_model "llama3.2:3b" "Llama 3.2 3B (general Q&A)"
pull_model "qwen2.5vl:3b" "Qwen2.5-VL 3B (OCR/vision)"

# Aryabhata needs special handling (not in Ollama registry)
info "Note: Aryabhata 1.0 (JEE Math) requires manual conversion from HuggingFace."
info "Run: python3 -m mlx_lm.convert --hf-path PhysicsWallahAI/Aryabhata-1.0 --mlx-path models/aryabhata-mlx-4bit -q"

# ── Step 5: Build Question Bank (qbg.db) ────────────────────
info "Step 5: Building question bank database..."

cd "$BASE_DIR"
source venv/bin/activate

if [ -f "data/qbg.db" ]; then
    SIZE=$(du -h data/qbg.db | awk '{print $1}')
    log "qbg.db exists ($SIZE)"
else
    warn "Building qbg.db from 3 HuggingFace datasets (~1.14M questions)..."
    warn "This downloads ~3GB and takes 10-15 minutes."
    echo ""
    read -p "Build now? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python3 analysis/build_qbg_db.py
        log "qbg.db built"
    else
        warn "Skipped. Build later with: python3 analysis/build_qbg_db.py"
    fi
fi

# Label topics
if [ -f "data/qbg.db" ]; then
    LABELED=$(python3 -c "import sqlite3; c=sqlite3.connect('data/qbg.db'); print(c.execute('SELECT COUNT(*) FROM questions WHERE topic IS NOT NULL').fetchone()[0])" 2>/dev/null || echo "0")
    if [ "$LABELED" -gt 0 ]; then
        log "Topic labels: $LABELED questions labeled"
    else
        info "Labeling topics..."
        python3 analysis/label_qbg_topics.py 2>/dev/null || warn "Topic labeling skipped (run manually)"
    fi
fi

# ── Step 6: Portal setup ─────────────────────────────────────
info "Step 6: Setting up Next.js portal..."

if [ -d "$PORTAL_DIR" ]; then
    cd "$PORTAL_DIR"
    if [ ! -d "node_modules" ]; then
        npm install --silent
        log "Portal dependencies installed"
    else
        log "Portal dependencies exist"
    fi
else
    warn "Portal not found at $PORTAL_DIR"
    warn "Clone it: git clone https://github.com/firnweh/prajna-portal.git $PORTAL_DIR"
fi

# Create .env.local
if [ -d "$PORTAL_DIR" ] && [ ! -f "$PORTAL_DIR/.env.local" ]; then
    cat > "$PORTAL_DIR/.env.local" << 'ENVEOF'
NEXT_PUBLIC_BACKEND_URL=http://localhost:4000
NEXT_PUBLIC_INTEL_URL=http://localhost:8001
ENVEOF
    log "Created portal .env.local"
fi

# ── Step 7: Start all services ───────────────────────────────
info "Step 7: Starting services..."

cd "$BASE_DIR"

# Stop existing pm2 processes
pm2 delete prajna prajna-intelligence 2>/dev/null || true

# Start Node.js backend
pm2 start backend/server.js --name prajna --cwd "$BASE_DIR/backend" 2>/dev/null
log "Backend started (port 4000)"

# Start Intelligence API
pm2 start "source $BASE_DIR/venv/bin/activate && cd $BASE_DIR/intelligence && uvicorn services.api.main:app --host 0.0.0.0 --port 8001" --name prajna-intelligence --interpreter bash 2>/dev/null
log "Intelligence API started (port 8001)"

# Wait for services
sleep 5

# ── Step 8: Verify ───────────────────────────────────────────
info "Step 8: Verifying setup..."

check_service() {
    local url=$1
    local name=$2
    local code=$(curl -s -o /dev/null -w "%{http_code}" "$url" 2>/dev/null || echo "000")
    if [ "$code" = "200" ] || [ "$code" = "307" ]; then
        log "$name: ✓ running ($code)"
    else
        err "$name: ✗ not responding ($code)"
    fi
}

check_service "http://localhost:4000/api/auth/health" "Node.js Backend"
check_service "http://localhost:8001/docs" "Intelligence API"
check_service "http://localhost:11434/api/tags" "Ollama"

# Check models
echo ""
info "Ollama models:"
curl -s http://localhost:11434/api/tags 2>/dev/null | python3 -c "
import sys, json
d = json.load(sys.stdin)
for m in d.get('models', []):
    print(f'  ✓ {m[\"name\"]} ({m[\"size\"]/1e9:.1f} GB)')
" 2>/dev/null

# ── Done ─────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  ✅ PRAJNA Setup Complete!                               ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║                                                          ║"
echo "║  Start the portal:                                       ║"
echo "║  cd $PORTAL_DIR"
echo "║  npm run dev -- --port 3000                              ║"
echo "║                                                          ║"
echo "║  URLs:                                                   ║"
echo "║  Portal:    http://localhost:3000                         ║"
echo "║  Backend:   http://localhost:4000                         ║"
echo "║  Intel API: http://localhost:8001/docs                    ║"
echo "║  Ollama:    http://localhost:11434                        ║"
echo "║                                                          ║"
echo "║  Login:                                                  ║"
echo "║  Student: 22300192@pw.live / prajna@2025                 ║"
echo "║  Admin:   admin@prajna.ai / prajna@2025                  ║"
echo "║                                                          ║"
echo "║  Manage services: pm2 list | pm2 logs | pm2 restart all  ║"
echo "╚══════════════════════════════════════════════════════════╝"

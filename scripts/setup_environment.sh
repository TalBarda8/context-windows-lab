#!/bin/bash

# Setup script for Context Windows Lab
# This script sets up the environment and verifies all prerequisites

set -e  # Exit on error

echo "========================================="
echo "Context Windows Lab - Environment Setup"
echo "========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print success messages
print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

# Function to print warning messages
print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Function to print error messages
print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Step 1: Check Python version
echo "[1/6] Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    print_success "Python $PYTHON_VERSION found"
else
    print_error "Python 3 not found. Please install Python 3.10 or higher."
    exit 1
fi

# Step 2: Create virtual environment
echo ""
echo "[2/6] Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_warning "Virtual environment already exists"
fi

# Step 3: Activate virtual environment and install dependencies
echo ""
echo "[3/6] Installing dependencies..."
source venv/bin/activate
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet

print_success "Dependencies installed"

# Step 4: Check Ollama installation
echo ""
echo "[4/6] Checking Ollama installation..."
if command -v ollama &> /dev/null; then
    print_success "Ollama found"

    # Check if Ollama is running
    if curl -s http://localhost:11434/api/tags &> /dev/null; then
        print_success "Ollama service is running"

        # List available models
        echo ""
        echo "Available models:"
        ollama list

        # Check for required model
        if ollama list | grep -q "llama3.2"; then
            print_success "llama3.2 model found"
        else
            print_warning "llama3.2 model not found"
            echo "Pulling llama3.2 model (this may take a few minutes)..."
            ollama pull llama3.2
            print_success "llama3.2 model downloaded"
        fi
    else
        print_warning "Ollama service not running"
        echo "Please start Ollama with: ollama serve"
    fi
else
    print_error "Ollama not found"
    echo "Please install Ollama from: https://ollama.ai"
    exit 1
fi

# Step 5: Create necessary directories
echo ""
echo "[5/6] Creating project directories..."
mkdir -p data/synthetic/needle_haystack
mkdir -p data/synthetic/context_size
mkdir -p data/hebrew_corpus
mkdir -p results/{exp1,exp2,exp3,exp4}
mkdir -p notebooks

print_success "Directories created"

# Step 6: Verify installation
echo ""
echo "[6/6] Verifying installation..."
python3 -c "import langchain; import chromadb; import sentence_transformers; print('All packages imported successfully')"
print_success "All packages verified"

echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "  1. Activate virtual environment: source venv/bin/activate"
echo "  2. Run all experiments: bash scripts/run_all_experiments.sh"
echo "  3. Or run individual experiments:"
echo "     - python -m src.experiments.exp1_needle_haystack"
echo "     - python -m src.experiments.exp2_context_size"
echo "     - python -m src.experiments.exp3_rag_impact"
echo "     - python -m src.experiments.exp4_strategies"
echo ""

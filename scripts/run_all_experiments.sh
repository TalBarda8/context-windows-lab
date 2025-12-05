#!/bin/bash

# Script to run all Context Windows Lab experiments
# This script runs all four experiments in sequence

set -e  # Exit on error

echo "================================================================="
echo "Context Windows Lab - Running All Experiments"
echo "================================================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print section headers
print_header() {
    echo -e "\n${BLUE}=================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}=================================================================${NC}\n"
}

# Function to print success messages
print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

# Function to print error messages
print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${YELLOW}⚠${NC} Virtual environment not activated"
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Check if Ollama is running
echo "Checking prerequisites..."
if ! curl -s http://localhost:11434/api/tags &> /dev/null; then
    print_error "Ollama service is not running"
    echo "Please start Ollama with: ollama serve"
    exit 1
fi
print_success "Ollama service is running"

# Start time
START_TIME=$(date +%s)

# Experiment 1: Needle in Haystack
print_header "Experiment 1: Needle in Haystack (Lost in the Middle)"
python3 -m src.experiments.exp1_needle_haystack

if [ $? -eq 0 ]; then
    print_success "Experiment 1 completed successfully"
else
    print_error "Experiment 1 failed"
    exit 1
fi

# Experiment 2: Context Window Size Impact
print_header "Experiment 2: Context Window Size Impact"
python3 -m src.experiments.exp2_context_size

if [ $? -eq 0 ]; then
    print_success "Experiment 2 completed successfully"
else
    print_error "Experiment 2 failed"
    exit 1
fi

# Experiment 3: RAG Impact
print_header "Experiment 3: RAG Impact"
python3 -m src.experiments.exp3_rag_impact

if [ $? -eq 0 ]; then
    print_success "Experiment 3 completed successfully"
else
    print_error "Experiment 3 failed"
    exit 1
fi

# Experiment 4: Context Engineering Strategies
print_header "Experiment 4: Context Engineering Strategies"
python3 -m src.experiments.exp4_strategies

if [ $? -eq 0 ]; then
    print_success "Experiment 4 completed successfully"
else
    print_error "Experiment 4 failed"
    exit 1
fi

# Calculate total time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

# Final summary
print_header "All Experiments Complete!"

echo ""
echo "Execution Summary:"
echo "  - Total time: ${MINUTES}m ${SECONDS}s"
echo ""
echo "Results saved to:"
echo "  - results/exp1/ - Needle in Haystack results"
echo "  - results/exp2/ - Context Size Impact results"
echo "  - results/exp3/ - RAG Impact results"
echo "  - results/exp4/ - Context Engineering results"
echo ""
echo "Next steps:"
echo "  1. Review results in the results/ directory"
echo "  2. Open Jupyter notebooks for detailed analysis"
echo "  3. Check visualizations (PNG files in results/)"
echo ""

print_success "All experiments completed successfully!"
echo ""

#!/bin/bash
# Script d'initialisation Git pour Unified XAI Platform
# Usage: ./setup_git.sh YOUR_GITHUB_USERNAME

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
print_step() {
    echo -e "${GREEN}==>${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if username provided
if [ -z "$1" ]; then
    print_error "Usage: ./setup_git.sh YOUR_GITHUB_USERNAME"
    exit 1
fi

GITHUB_USERNAME=$1
REPO_NAME="unified-xai-platform"

print_step "🚀 Initializing Git for Unified XAI Platform"
echo ""

# Step 1: Clean Python cache
print_step "Step 1/8: Cleaning Python cache files..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true
find . -type f -name ".DS_Store" -delete 2>/dev/null || true
echo "✅ Cleaned"
echo ""

# Step 2: Check if .gitignore exists
print_step "Step 2/8: Checking .gitignore..."
if [ ! -f ".gitignore" ]; then
    print_warning ".gitignore not found. Please copy the provided .gitignore file."
    exit 1
fi
echo "✅ .gitignore exists"
echo ""

# Step 3: Create models directory structure
print_step "Step 3/8: Setting up models directory..."
mkdir -p models/audio/huggingface
mkdir -p models/audio/tensorflow
touch models/.gitkeep

cat > models/README.md << 'EOF'
# Models Directory

⚠️ **Important**: Model files are NOT included in Git (too large).

## Image Models
- Automatically downloaded by torchxrayvision on first run

## Audio Models
- Place wav2vec2 models in: `models/audio/huggingface/`
- Place TensorFlow CNN in: `models/audio/tensorflow/`

## Structure
```
models/
├── .gitkeep
├── README.md
├── audio/
│   ├── huggingface/
│   └── tensorflow/
└── image/
    └── [auto-downloaded]
```
EOF

echo "✅ Models directory configured"
echo ""

# Step 4: Initialize Git
print_step "Step 4/8: Initializing Git repository..."
if [ -d ".git" ]; then
    print_warning "Git already initialized. Skipping..."
else
    git init
    echo "✅ Git initialized"
fi
echo ""

# Step 5: Add all files
print_step "Step 5/8: Adding files to Git..."
git add .
echo "✅ Files staged"
echo ""

# Step 6: First commit
print_step "Step 6/8: Creating initial commit..."
git commit -m "Initial commit: Unified XAI Platform v1.0

- Multi-modal XAI system (images + audio)
- 7 XAI methods (LIME, Grad-CAM, SHAP, etc.)
- 9 pre-trained models support
- 4 Streamlit pages
- Complete documentation
- Production-ready code"
echo "✅ Initial commit created"
echo ""

# Step 7: Set up remote
print_step "Step 7/8: Configuring GitHub remote..."
REMOTE_URL="https://github.com/$GITHUB_USERNAME/$REPO_NAME.git"

# Check if remote already exists
if git remote | grep -q "origin"; then
    print_warning "Remote 'origin' already exists. Updating URL..."
    git remote set-url origin $REMOTE_URL
else
    git remote add origin $REMOTE_URL
fi

# Rename branch to main
git branch -M main
echo "✅ Remote configured: $REMOTE_URL"
echo ""

# Step 8: Instructions
print_step "Step 8/8: Final steps"
echo ""
echo "🎉 Git setup complete!"
echo ""
echo "📋 Next steps:"
echo ""
echo "1. Create repository on GitHub:"
echo "   - Go to: https://github.com/new"
echo "   - Name: $REPO_NAME"
echo "   - Description: Multi-modal Explainable AI platform"
echo "   - Public or Private (your choice)"
echo "   - DO NOT initialize with README, .gitignore, or license"
echo "   - Click 'Create repository'"
echo ""
echo "2. Push to GitHub:"
echo "   ${GREEN}git push -u origin main${NC}"
echo ""
echo "3. If prompted for authentication:"
echo "   - Username: $GITHUB_USERNAME"
echo "   - Password: Your Personal Access Token (NOT your password)"
echo "   - Create token: GitHub → Settings → Developer settings → Tokens"
echo ""
echo "4. Verify on GitHub:"
echo "   https://github.com/$GITHUB_USERNAME/$REPO_NAME"
echo ""
echo "📚 For detailed instructions, see GUIDE_GIT.md"
echo ""
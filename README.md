# IVF-CE: Inverted File Index with Cross-Cluster Exploration

## Team Members
- Tameer Milhem (tameermilhem@campus.technion.ac.il)
- Luay Marzok (luay.m@campus.technion.ac.il)
- Rany Khirbawi (ranykhirbawi@campus.technion.ac.il)
- Weaam Mulla (weaam.mulla@campus.technion.ac.il)

## Overview
IVF with Cross-Cluster Exploration for Approximate Nearest Neighbor Search.

## Installation
```bash
git clone https://github.com/tmeermilhem/ivf-ce-ann.git
cd ivf-ce-ann
pip install -r requirements.txt
Structure

config/ - Configuration files
data/ - Dataset utilities
src/ - Source code
evaluation/ - Metrics
experiments/ - Experiment scripts
results/ - Output files EOF
Code

### Create `.gitignore`:
```bash
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*.so
.Python
venv/
env/
*.egg-info/

# Data
data/*.fvecs
data/*.ivecs
data/*.npy
data/raw/
data/processed/

# Results
results/logs/*.log
*.index
*.faiss

# IDE
.vscode/
.idea/
.DS_Store

# Jupyter
.ipynb_checkpoints/

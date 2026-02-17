#!/bin/bash
# Quick dashboard update script

cd "$(dirname "$0")"

echo "📊 Regenerating dashboard..."
/home/codespace/.python/current/bin/python -c "
from pathlib import Path
import sys
sys.path.insert(0, 'src')
from dashboard_generator import DashboardGenerator

generator = DashboardGenerator(Path('.'))
success = generator.generate(Path('docs/index.html'))
print('✅ Dashboard updated!' if success else '❌ Failed')
"

if [ "$1" == "--push" ]; then
    echo "🚀 Pushing to GitHub..."
    git add docs/index.html
    git commit -m "Manual dashboard update $(date '+%Y-%m-%d %H:%M')" 2>/dev/null && \
    git push origin main && \
    echo "✅ Pushed to GitHub Pages" || \
    echo "⚠️  No changes to push"
fi

echo ""
echo "View locally: file://$(pwd)/docs/index.html"
echo "View on GitHub Pages: https://shimmy-shams.github.io/Quant/"

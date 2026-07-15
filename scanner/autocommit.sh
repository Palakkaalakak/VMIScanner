#!/bin/bash
# Auto-commit AND PUSH all uncommitted work every 60s. Session wipes restore
# from GitHub, so nothing can be lost anymore.
cd /home/user/webapp
while true; do
  sleep 60
  if [ -n "$(git status --porcelain 2>/dev/null)" ]; then
    git add -A
    git commit -m "autosave $(date -u +%H:%M:%SZ)" --quiet 2>/dev/null
    git push origin main --quiet 2>/dev/null
  fi
done

#!/bin/bash
# Double-click in Finder to open the cupbot live dashboard in your browser.
cd "$(dirname "$0")"
if ! curl -s -o /dev/null http://127.0.0.1:8790/api/state; then
  nohup nice -n 15 python3 -m cupbot dashboard > journal/logs/dashboard.log 2>&1 &
  sleep 1
fi
open http://127.0.0.1:8790

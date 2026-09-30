#!/usr/bin/env bash
# From the Mac: copy the code to the server and restart the bot.
#     bash deploy/push.sh ubuntu@<server-ip> [path/to/key.pem]
# The server's .env, journal and recorded data are never overwritten.
set -euo pipefail
HOST="$1"; KEY="${2:-}"
SSH="ssh ${KEY:+-i $KEY}"
cd "$(dirname "$0")/.."
rsync -az -e "$SSH" --exclude .venv --exclude .env --exclude journal --exclude 'tuning/data' \
      --exclude __pycache__ --exclude .git ./ "$HOST:Trading/"
$SSH "$HOST" 'sudo rsync -a --exclude .venv --exclude .env --exclude journal --exclude tuning/data ~/Trading/ /home/cupbot/Trading/ \
  && sudo chown -R cupbot:cupbot /home/cupbot/Trading \
  && sudo -u cupbot /home/cupbot/Trading/.venv/bin/pip install -q -r /home/cupbot/Trading/requirements.txt \
  && (systemctl is-active --quiet cupbot && sudo systemctl restart cupbot || true) \
  && sudo systemctl restart cupbot-recorder && echo pushed'

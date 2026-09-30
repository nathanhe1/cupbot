#!/usr/bin/env bash
# One-time setup on a fresh Ubuntu 24.04 server (AWS Lightsail or EC2, region us-east-1).
# Run as the default user (ubuntu) after copying the Trading folder to the server:
#     bash ~/Trading/deploy/setup_vm.sh
set -euo pipefail

sudo apt-get update -y
sudo apt-get install -y python3 python3-venv python3-pip chrony sqlite3 git
sudo systemctl enable --now chrony          # accurate clock: needed to measure the Cup's lag

# dedicated user; the code lives in /home/cupbot/Trading
id cupbot >/dev/null 2>&1 || sudo useradd -m -s /bin/bash cupbot
sudo rsync -a --delete --exclude .venv --exclude journal --exclude 'tuning/data' "$HOME/Trading/" /home/cupbot/Trading/
sudo mkdir -p /home/cupbot/Trading/journal
[ -f /home/cupbot/Trading/.env ] || sudo cp /home/cupbot/Trading/.env.example /home/cupbot/Trading/.env
sudo chmod 600 /home/cupbot/Trading/.env
sudo chown -R cupbot:cupbot /home/cupbot/Trading

sudo -u cupbot python3 -m venv /home/cupbot/Trading/.venv
sudo -u cupbot /home/cupbot/Trading/.venv/bin/pip install -q --upgrade pip
sudo -u cupbot /home/cupbot/Trading/.venv/bin/pip install -q -r /home/cupbot/Trading/requirements.txt

# keep system logs bounded (the bot's own journal is a small SQLite file)
sudo mkdir -p /etc/systemd/journald.conf.d
printf "[Journal]\nSystemMaxUse=500M\n" | sudo tee /etc/systemd/journald.conf.d/cupbot.conf >/dev/null
sudo systemctl restart systemd-journald

sudo cp /home/cupbot/Trading/deploy/cupbot.service /home/cupbot/Trading/deploy/cupbot-recorder.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now cupbot-recorder
sudo systemctl enable cupbot        # starts on boot; start it yourself once .env is filled in

# `cupbot` shortcut for SSH sessions (e.g. from a phone): cupbot kill / status / logs
sudo tee /usr/local/bin/cupbot >/dev/null <<'SH'
#!/bin/sh
cd /home/cupbot/Trading
case "$1" in
  logs)    exec journalctl -u cupbot -f -n 100 ;;
  status)  systemctl --no-pager status cupbot | head -5; exec sudo -u cupbot .venv/bin/python -m cupbot status ;;
  restart) exec sudo systemctl restart cupbot ;;
  stop)    exec sudo systemctl stop cupbot ;;
  *)       exec sudo -u cupbot .venv/bin/python -m cupbot "$@" ;;
esac
SH
sudo chmod +x /usr/local/bin/cupbot

echo
echo "Done. Next:"
echo "  1. sudo nano /home/cupbot/Trading/.env      (API key, NTFY_TOPIC, NTFY_CMD_TOKEN)"
echo "  2. cupbot preflight"
echo "  3. sudo systemctl start cupbot && cupbot logs   (dry run until CUPBOT_ARGS=--live)"

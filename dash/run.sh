#!/bin/bash
# Keeps the dashboard server up (restarts it if it exits). Stop with: kill $(cat run.pid) $(cat server.pid)
cd "$(dirname "$0")"
echo $$ > run.pid
while true; do
  python3 server.py --port 8771 >> server.log 2>&1 &
  echo $! > server.pid; wait $!
  echo "$(date) server exited ($?), restarting" >> server.log; sleep 3
done

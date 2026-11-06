#!/bin/bash
# Refresh the data for the pinned "Hidden-state preference" dashboard: pull the zones logs from arena9-trouble now
# (instead of waiting for dash/server.py's 2-minute sync), rebuild runs/zones/dash/*.json, and wrap the page source as a
# store document. Claude then uploads the five JSON files and points the datasets at them.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
BASE=/root/Dropbox/ARENA/weight_space_prog/runs/zones
rsync -a --include='*/' --include='*.json' --include='margins.jsonl' --exclude='*' \
  -e "ssh -o BatchMode=yes -o ConnectTimeout=10" arena9-trouble:/root/wsp-dq/runs/zones/ $BASE/ 2>&1 | grep -v Warn || true
python3 $HERE/zones_dash_build.py --base $BASE
python3 -c "import json,datetime,sys; json.dump({'text': open('$HERE/runs/zones_dash/index.html').read(), 'edit': {'via': 'claude', 'at': datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}}, open('$BASE/dash/doc_index.json','w'))"
echo refreshed

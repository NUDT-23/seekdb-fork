#!/bin/bash
# Restart seekdb with the freshly built binary and run the ai_function test.
for p in $(pgrep -f "observer/seekdb"); do kill -9 "$p" 2>/dev/null; done
sleep 4
rm -rf /workspace/data/log
(setsid nohup /workspace/start_obs.sh > /workspace/obs.log 2>&1 &)
for i in $(seq 1 30); do
  if mysql -h127.0.0.1 -P2881 -uroot -e "select 1" >/dev/null 2>&1; then break; fi
  sleep 2
done
mysql -h127.0.0.1 -P2881 -uroot -e "create database if not exists test" 2>/dev/null
cd /workspace || exit 1
/opt/venv/bin/python3 /workspace/run_ai_test.py 2881 > /workspace/aitest2.log 2>&1
echo "TEST_DONE rc=$?"

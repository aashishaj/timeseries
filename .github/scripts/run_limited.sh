#!/usr/bin/env bash
# run_limited.sh SECONDS COMMAND [ARGS...]
#
# Runs COMMAND. If it is still running after SECONDS, prints where every thread of
# it (and of its child processes) is stuck, then stops it:
#   * native stacks with macOS `sample` (shows hangs inside C libraries),
#   * Python stacks via SIGABRT, if PYTHONFAULTHANDLER=1 is set.
# Exit status: COMMAND's own, or 124 if it had to be stopped.
set -u
limit=$1
shift
start=$(date +%s)
"$@" &
pid=$!
while kill -0 "$pid" 2>/dev/null; do
  if [ $(( $(date +%s) - start )) -ge "$limit" ]; then
    echo "::warning::still running after ${limit}s: $*"
    procs=$(echo "$pid" $(pgrep -P "$pid"))
    ps -o pid,ppid,stat,etime,%cpu,command -p "$(echo $procs | tr ' ' ',')" 2>/dev/null | cut -c1-200
    for p in $procs; do
      echo "----- native stacks of pid $p (sample) -----"
      sample "$p" 2 -mayDie 2>/dev/null | sed -n '/^Call graph:/,/^Total number in stack/p' | cut -c1-160 | head -120
    done
    echo "----- Python stacks (faulthandler) -----"
    for p in $procs; do kill -ABRT "$p" 2>/dev/null; done
    sleep 3
    for p in $procs; do kill -9 "$p" 2>/dev/null; done
    wait "$pid" 2>/dev/null
    echo "stopped after $(( $(date +%s) - start ))s: $*"
    exit 124
  fi
  sleep 1
done
wait "$pid"
status=$?
echo "finished in $(( $(date +%s) - start ))s with exit $status: $*"
exit "$status"

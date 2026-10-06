#!/bin/bash
# Pi health log for load tests: every 2 s append the time, uptime, CPU temperature, ARM clock, `get_throttled` flags
# and load average to <log>, synced to disk so the last lines survive a reset (the journal is not persistent).
# Independent of the guard inside pi/benchmark.py and pi/app.py, it applies the same under-voltage rule: when the
# last 60 s hold <max seconds> of under-voltage (default 10) or <max dips> separate dips (default 3), it stops both
# with SIGTERM. Start it in the background next to the job and kill it afterwards:
#     bash pi/watch.sh results/run.watch.log &
# usage: watch.sh <log> [max seconds] [max dips]
LOG=$1 MAX_S=${2:-10} MAX_DIPS=${3:-3}
EVERY=2 WINDOW=60
flags=()  # under-voltage now (1/0) of the last WINDOW / EVERY samples
while true; do
    t=$(vcgencmd get_throttled | cut -d= -f2)
    echo "$(date +%T) up=$(cut -d' ' -f1 /proc/uptime) temp=$(vcgencmd measure_temp | cut -d= -f2) arm=$(vcgencmd measure_clock arm | cut -d= -f2) throttled=$t load=$(cut -d' ' -f1 /proc/loadavg)" >> "$LOG"
    flags+=($(( t & 1 )))
    (( ${#flags[@]} > WINDOW / EVERY )) && flags=("${flags[@]:1}")
    uv=0 dips=0 previous=0
    for flag in "${flags[@]}"; do
        (( uv += flag, dips += flag && !previous, previous = flag ))
    done
    if (( (MAX_S > 0 && uv * EVERY >= MAX_S) || (MAX_DIPS > 0 && dips >= MAX_DIPS) )); then
        echo "$(date +%T) ABORT: under-voltage for $((uv * EVERY)) s in $dips dip(s) in the last $WINDOW s, stopping pi/benchmark.py and pi/app.py" >> "$LOG"
        pkill -TERM -f '[p]i/(benchmark|app)\.py'
        flags=()
    fi
    sync "$LOG"
    sleep $EVERY
done

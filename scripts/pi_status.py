"""One-shot status of the Raspberry Pi: one SSH login plus one HTTP request to the app (logins cost the Pi CPU).

    python scripts/pi_status.py
    python scripts/pi_status.py --log 30      # more lines of the app's service log

Prints the boot time (compare it with an earlier one to spot a reset; the user also powers the Pi off to move it), the
cats-app service state, CPU temperature, ARM clock, the decoded `vcgencmd get_throttled` flags, free disk, the captures
per day, the end of results/phase6/service.log and, if the app answers, its readiness and detection rate from /stats.
Exit code 1 when the Pi can't be reached.
"""
import argparse
import json
import sys
import urllib.request

from pi_remote import Pi, load_config

FLAGS = {0: "under-voltage", 1: "ARM frequency capped", 2: "throttled", 3: "soft temperature limit"}
STATUS = r"""
cd ~/cats-localization-v2 || exit 1
echo "boot=$(uptime -s)"
echo "now=$(date '+%F %T %Z')"
echo "service=$(systemctl is-active cats-app) / $(systemctl is-enabled cats-app 2>/dev/null)"
systemctl show cats-app -p ActiveEnterTimestamp -p NRestarts -p ExecMainStatus | sed 's/^/service_/'
echo "throttled=$(vcgencmd get_throttled | cut -d= -f2)"
echo "temp=$(vcgencmd measure_temp | cut -d= -f2)"
echo "arm_mhz=$(( $(vcgencmd measure_clock arm | cut -d= -f2) / 1000000 ))"
echo "disk_free=$(df -h --output=avail / | tail -1 | tr -d ' ')"
for d in captures/*/; do [ -d "$d" ] && echo "captures $(basename "$d")=$(ls "$d" | grep -c 'jpg$')"; done
echo "log_begin"
grep -v ' INFO ' results/phase6/service.log 2>/dev/null | tail -n {log}
"""


def decode_throttled(raw: str) -> str:
    value = int(raw, 16)
    now = [name for bit, name in FLAGS.items() if value >> bit & 1]
    since = [name for bit, name in FLAGS.items() if value >> (bit + 16) & 1]
    return f"{raw} (now: {', '.join(now) or 'none'}; since boot: {', '.join(since) or 'none'})"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", type=int, default=8, help="lines of the service log to show")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # the log has non-ASCII characters (°C)

    config = load_config()
    try:
        with Pi(config) as pi:
            _, output = pi.run(STATUS.replace("{log}", str(args.log)), echo=False, timeout=60)
    except Exception as error:  # connection refused / timed out: usually powered off
        print(f"Pi {config['PI_HOST']} not reachable ({type(error).__name__}: {error}); is it powered on?")
        raise SystemExit(1)

    head, _, log = output.partition("log_begin\n")
    for line in head.splitlines():
        key, _, value = line.partition("=")
        if key == "throttled":
            value = decode_throttled(value)
        print(f"{key:28} {value}")
    print("service log (last lines, INFO lines left out):")
    for line in log.splitlines():
        print(f"  {line}")

    try:
        with urllib.request.urlopen(f"http://{config['PI_HOST']}:8000/stats", timeout=5) as response:
            stats = json.load(response)
    except OSError as error:
        print(f"app /stats: no answer ({error})")
        return
    captures, detection, system = stats["captures"], stats["detection"], stats["system"]
    window = system.get("undervoltage_window", {})
    print(f"app: up {stats['uptime_s'] / 60:.0f} min, ready {captures['ready']} (clock {captures['clock']}, "
          f"session {captures['session']}), {captures['saved']} captures this run, {captures['files']} on the Pi; "
          f"{detection['fps']:.1f} detections/s, {stats['stream']['viewers']} viewer(s); under-voltage in the last "
          f"{window.get('window_s', '?')} s: {window.get('seconds', '?')} s in {window.get('dips', '?')} dip(s)")


if __name__ == "__main__":
    main()

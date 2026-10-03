"""Soak-test sampler for pi/app.py: log its stats and memory over a long run, then summarise.

    python pi/soak.py --duration 2100 --out results/phase5/soak.jsonl    # sample, then print the summary
    python pi/soak.py --summarize runs/pi/phase5/soak.jsonl              # summary of an existing log (also on the PC)

Every --interval seconds it appends one JSON line, flushed and fsync'ed so the log survives a reset:
  - "stats": the app's /stats (FPS, stage times, detections, temperature, ARM clock, throttle flags), or "error";
  - "proc": its own /proc readings, independent of the app: VmRSS / VmHWM / threads and CPU use (% of one core since
    the previous sample) of the app, the detector process and any other child of the app (multiprocessing's
    resource tracker), plus the whole system's CPU busy % (of all cores), load average and MemAvailable.
Sampling ends after --duration seconds or when the app's process is gone.

The summary gives RSS at the start / end / max and its trend in MiB per hour (least squares over the samples after
--warmup seconds, to show whether memory leaks), plus temperature, clock, throttling, FPS and CPU use.
Standard library only: it adds almost no load to the Pi.
"""
import argparse
import json
import os
import statistics
import sys
import time
import urllib.request
from pathlib import Path

CLOCK_TICKS = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100


def read(path: str) -> str | None:
    try:
        return Path(path).read_text()
    except OSError:
        return None


def process_info(pid: int) -> dict | None:
    """VmRSS / VmHWM (MiB), threads, parent pid and CPU time (s) of a process; None if it is gone."""
    status, stat = read(f"/proc/{pid}/status"), read(f"/proc/{pid}/stat")
    if status is None or stat is None:
        return None
    values = dict(line.split(":", 1) for line in status.splitlines() if ":" in line)
    fields = stat.rsplit(")", 1)[1].split()  # after "pid (comm)": state ppid ... utime(12) stime(13)
    return {"rss": round(int(values["VmRSS"].split()[0]) / 1024, 1),
            "hwm": round(int(values["VmHWM"].split()[0]) / 1024, 1),
            "threads": int(values["Threads"]), "ppid": int(fields[1]),
            "cpu_s": (int(fields[11]) + int(fields[12])) / CLOCK_TICKS}


def children(pid: int) -> list[int]:
    found = []
    for entry in os.listdir("/proc"):
        if entry.isdigit():
            stat = read(f"/proc/{entry}/stat")
            if stat and int(stat.rsplit(")", 1)[1].split()[1]) == pid:
                found.append(int(entry))
    return found


def system_cpu() -> tuple[int, int]:
    """(busy, total) jiffies over all cores since boot."""
    values = [int(v) for v in read("/proc/stat").splitlines()[0].split()[1:]]
    idle = values[3] + values[4]  # idle + iowait
    return sum(values) - idle, sum(values)


def fetch_stats(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=5) as response:
        return json.load(response)


def sample_loop(args):
    args.out.parent.mkdir(parents=True, exist_ok=True)
    stats_url = args.url.rstrip("/") + "/stats"
    stats = fetch_stats(stats_url)  # fails here if the app is not running
    app_pid = stats["pids"]["app"]
    print(f"sampling app pid {app_pid} every {args.interval:.0f} s for {args.duration:.0f} s -> {args.out}",
          flush=True)

    previous_cpu, previous_system, previous_t = {}, system_cpu(), time.monotonic()
    start = time.monotonic()
    with args.out.open("a", encoding="utf-8") as out:
        while True:
            now = time.monotonic()
            record = {"t": round(now - start, 1), "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
            try:
                stats = fetch_stats(stats_url)
                record["stats"] = stats
            except Exception as error:  # noqa: BLE001 - any failure is part of the result
                stats = None
                record["error"] = f"{type(error).__name__}: {error}"

            app = process_info(app_pid)
            detector_pid = stats["pids"]["detector"] if stats else None
            procs = {}
            if app is not None:
                procs["app"] = {"pid": app_pid, **app}
                for pid in children(app_pid):
                    info = process_info(pid)
                    if info is not None:
                        procs["detector" if pid == detector_pid else f"child_{pid}"] = {"pid": pid, **info}
            elapsed = now - previous_t
            busy, total = system_cpu()
            for name, info in procs.items():
                key = info["pid"]
                if key in previous_cpu and elapsed > 0:
                    info["cpu_percent"] = round((info["cpu_s"] - previous_cpu[key]) / elapsed * 100, 1)
                previous_cpu[key] = info["cpu_s"]
            meminfo = dict(line.split(":", 1) for line in read("/proc/meminfo").splitlines())
            record["proc"] = {
                "processes": procs,
                "rss_total": round(sum(p["rss"] for p in procs.values()), 1),
                "system_cpu_percent": round((busy - previous_system[0]) / max(total - previous_system[1], 1)
                                            * 100, 1),
                "load": [float(v) for v in read("/proc/loadavg").split()[:3]],
                "mem_available": round(int(meminfo["MemAvailable"].split()[0]) / 1024, 1),
            }
            previous_system, previous_t = (busy, total), now

            out.write(json.dumps(record) + "\n")
            out.flush()
            os.fsync(out.fileno())

            if app is None:
                print(f"{record['time']} the app (pid {app_pid}) is gone, stopping", flush=True)
                break
            if now - start >= args.duration:
                break
            time.sleep(max(0.0, start + (int((now - start) / args.interval) + 1) * args.interval - time.monotonic()))


def slope_per_hour(t: list[float], values: list[float]) -> float | None:
    if len(t) < 3:
        return None
    return statistics.linear_regression(t, values).slope * 3600


def fmt(value, spec: str = ".1f") -> str:
    return "n/a" if value is None else format(value, spec)


def describe(values: list[float], spec: str = ".1f") -> str:
    if not values:
        return "n/a"
    return f"mean {statistics.fmean(values):{spec}}, min {min(values):{spec}}, max {max(values):{spec}}"


def summarize(path: Path, warmup: float):
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        sys.exit(f"{path}: no samples")
    ok = [r for r in records if "stats" in r]
    errors = len(records) - len(ok)
    duration = records[-1]["t"]
    print(f"{path}: {len(records)} samples over {duration / 60:.1f} min ({records[0]['time']} -> "
          f"{records[-1]['time']}), {errors} without /stats")

    steady = [r for r in records if r["t"] >= warmup and "app" in r["proc"]["processes"]]
    print(f"\nmemory (MiB, /proc; trend over t >= {warmup:.0f} s, {len(steady)} samples):")
    names = sorted({name for r in records for name in r["proc"]["processes"]}, key=lambda n: (n != "app", n))
    for name in names + ["total"]:
        rows = [(r["t"], r["proc"]["rss_total"] if name == "total" else r["proc"]["processes"][name]["rss"])
                for r in records if name == "total" or name in r["proc"]["processes"]]
        trend = [(t, v) for t, v in rows if t >= warmup]
        slope = slope_per_hour([t for t, _ in trend], [v for _, v in trend])
        hwm = [r["proc"]["processes"][name]["hwm"] for r in records if name in r["proc"]["processes"]]
        print(f"  {name:14} RSS first {rows[0][1]:6.1f}  after warm-up {fmt(trend[0][1] if trend else None):>6}"
              f"  last {rows[-1][1]:6.1f}  max {max(v for _, v in rows):6.1f}  trend {fmt(slope, '+.2f'):>6} MiB/h"
              + (f"  peak (VmHWM) {hwm[-1]:.1f}" if hwm else ""))
    available = [r["proc"]["mem_available"] for r in records]
    print(f"  MemAvailable   {describe(available)}")

    def stat(*keys):
        values = []
        for r in ok:
            value = r["stats"]
            for key in keys:
                value = value.get(key) if isinstance(value, dict) else None
            if value is not None:
                values.append(value)
        return values

    print("\nperformance (/stats, 10 s windows):")
    print(f"  capture FPS    {describe(stat('capture', 'fps'), '.2f')}")
    print(f"  detection FPS  {describe(stat('detection', 'fps'), '.2f')}")
    print(f"  stream FPS     {describe(stat('stream', 'fps'), '.2f')}")
    print(f"  viewers        {describe(stat('stream', 'viewers'), '.0f')}")
    for key in ("preprocess", "infer", "postprocess", "roundtrip", "latency"):
        print(f"  {key + ' ms':14} {describe(stat('detection', 'mean_ms', key))}")
    print(f"  render ms      {describe(stat('stream', 'render_ms'))}")

    print("\nCPU (% of one core, /proc):")
    for name in names:
        values = [r["proc"]["processes"][name].get("cpu_percent") for r in records
                  if name in r["proc"]["processes"]]
        print(f"  {name:14} {describe([v for v in values if v is not None])}")
    print(f"  system busy    {describe([r['proc']['system_cpu_percent'] for r in records[1:]])} (% of all 4 cores)")

    temps, clocks = stat("system", "temp_c"), stat("system", "arm_mhz")
    raws = [r["stats"]["system"]["throttled"]["raw"] for r in ok if r["stats"]["system"].get("throttled")]
    flagged = [r for r in ok if r["stats"]["system"].get("throttled", {}).get("now")]
    print("\nPi health:")
    print(f"  temperature °C {describe(temps)}; first {fmt(temps[0] if temps else None)}, "
          f"last {fmt(temps[-1] if temps else None)}")
    print(f"  ARM clock MHz  {describe(clocks, '.0f')}; samples below 1200: {sum(c < 1200 for c in clocks)}"
          f" / {len(clocks)}")
    print(f"  throttle flags now in {len(flagged)} / {len(ok)} samples; values seen: {sorted(set(raws))}")
    last = ok[-1]["stats"]["system"] if ok else {}
    if last:
        print(f"  app's own 2-s samples: under-voltage {last['undervoltage_samples']}, capped "
              f"{last['capped_samples']} of {last['samples']}")
    if records[-1].get("error") or "app" not in records[-1]["proc"]["processes"]:
        print(f"\nlast sample: app {'gone' if 'app' not in records[-1]['proc']['processes'] else 'running'}, "
              f"{records[-1].get('error', '')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://localhost:8000", help="the app's address")
    parser.add_argument("--interval", type=float, default=5, help="seconds between samples")
    parser.add_argument("--duration", type=float, default=2100, help="seconds to sample")
    parser.add_argument("--warmup", type=float, default=300, help="seconds left out of the memory trend")
    parser.add_argument("--out", type=Path, default=Path("results/phase5/soak.jsonl"))
    parser.add_argument("--summarize", type=Path, metavar="LOG", help="only summarise an existing log")
    args = parser.parse_args()
    if args.summarize:
        summarize(args.summarize, args.warmup)
        return
    sample_loop(args)
    print()
    summarize(args.out, args.warmup)


if __name__ == "__main__":
    main()

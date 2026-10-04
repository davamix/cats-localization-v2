"""Test a push button on a GPIO pin. Read-only: the pin is only ever configured as an input.

    python pi/button_test.py                              # GPIO 25 (BCM), button to GND, internal pull-up, 50 ms debounce
    python pi/button_test.py --bounce-ms 0 --seconds 90   # raw edges, to see contact bounce

Prints the pin factory and the idle state, then one line per edge the pin reports (time, level, ms since the previous
edge, from the kernel's event timestamps) and gpiozero's press / release events, then a summary. Leave the button
alone for the first --quiet-s seconds: any edge in that window means a floating or noisy pin.

The pull must match the wiring: --pull up for a button between the pin and GND (pressed = low), --pull down for a
button between the pin and 3.3 V (pressed = high). Run it while pi/app.py is stopped if the app uses the same pin.
Needs gpiozero and lgpio (apt).
"""
import argparse
import signal
import threading
import time

from gpiozero import Button, Device


def now() -> str:
    t = time.time()
    return time.strftime("%H:%M:%S", time.localtime(t)) + f".{int(t % 1 * 1000):03d}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pin", type=int, default=25, help="GPIO number (BCM numbering, not the physical pin)")
    parser.add_argument("--pull", choices=["up", "down"], default="up",
                        help="internal pull: up for a button to GND, down for a button to 3.3 V")
    parser.add_argument("--bounce-ms", type=float, default=50, help="debounce time in ms (0 = off: raw edges)")
    parser.add_argument("--seconds", type=float, default=90, help="how long to listen")
    parser.add_argument("--quiet-s", type=float, default=20,
                        help="first seconds in which nobody touches the button (floating-pin check)")
    args = parser.parse_args()

    button = Button(args.pin, pull_up=args.pull == "up", bounce_time=args.bounce_ms / 1000 or None)
    start = time.monotonic()
    edges = []  # (seconds since start, kernel event time in s, level)
    events = {"pressed": 0, "released": 0}
    lock = threading.Lock()

    # Log every edge the pin reports, then pass it on to the Button. gpiozero keeps only a weak reference to the
    # callback, so `on_edge` must stay referenced while the pin is open (it does: it is a local of main()).
    button_handler = button.pin.when_changed

    def on_edge(ticks: float, level: int):
        with lock:
            gap = f"{(ticks - edges[-1][1]) * 1000:9.1f} ms" if edges else "    first edge"
            edges.append((time.monotonic() - start, ticks, level))
            pressed = level == (0 if args.pull == "up" else 1)
            print(f"{now()} edge {'low ' if level == 0 else 'high'} ({'pressed' if pressed else 'released'}) {gap}",
                  flush=True)
        button_handler(ticks, level)

    button.pin.when_changed = on_edge

    def on_event(name: str):
        def handler():
            with lock:
                events[name] += 1
                print(f"{now()}   Button: {name.upper()} (#{events[name]})", flush=True)
        return handler

    button.when_pressed, button.when_released = on_event("pressed"), on_event("released")

    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())
    try:
        print(f"pin factory {type(Device.pin_factory).__name__}, GPIO {args.pin}, pull-{args.pull}, "
              f"debounce {args.bounce_ms:g} ms", flush=True)
        print(f"idle: level {'high' if button.pin.state else 'low'}, is_pressed {button.is_pressed}", flush=True)
        print(f"{now()} listening for {args.seconds:g} s: don't touch the button for the first {args.quiet_s:g} s",
              flush=True)
        if not stop.wait(args.quiet_s):
            print(f"{now()} quiet window over: press now", flush=True)
            stop.wait(args.seconds - args.quiet_s)
        level_at_end = button.pin.state
    finally:
        button.close()  # releases the pin (it stays an input)

    with lock:
        quiet = [e for e in edges if e[0] < args.quiet_s]
        gaps = [(b[1] - a[1]) * 1000 for a, b in zip(edges, edges[1:])]
        # press = first "pressed" edge after a release -> first "released" edge after it (bounce edges in between)
        press_lengths, pressed_at = [], None
        for _, ticks, level in edges:
            if level == (0 if args.pull == "up" else 1):
                pressed_at = ticks if pressed_at is None else pressed_at
            elif pressed_at is not None:
                press_lengths.append((ticks - pressed_at) * 1000)
                pressed_at = None
    print("summary:")
    print(f"  edges {len(edges)}, of which {len(quiet)} in the quiet window"
          f"{' -> FLOATING OR NOISY PIN' if quiet else ' (pin stable when untouched)'}")
    print(f"  Button events: {events['pressed']} pressed, {events['released']} released")
    if gaps:
        short = sum(g < 20 for g in gaps)
        print(f"  shortest gap between edges {min(gaps):.2f} ms; {short} gap(s) < 20 ms"
              f"{' -> contact bounce' if short else ''}")
    if press_lengths:
        print(f"  press-to-release times: {', '.join(f'{p:.0f}' for p in press_lengths)} ms")
    print(f"  level at the end: {'high' if level_at_end else 'low'}")


if __name__ == "__main__":
    main()

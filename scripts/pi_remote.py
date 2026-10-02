"""Run commands and copy files on the Raspberry Pi over SSH (password auth).

Connection settings come from `pi.env` in the repository root (see `pi.env.example`); environment variables with
the same names take precedence.

Usage:
    python scripts/pi_remote.py run "uname -a"
    python scripts/pi_remote.py run --sudo "apt-get update"
    python scripts/pi_remote.py put pi ~/cats-localization-v2/pi        # file or folder, recursive
    python scripts/pi_remote.py get ~/capture.jpg captures/capture.jpg
"""
import argparse
import os
import posixpath
import shlex
import stat
import sys
from pathlib import Path

import paramiko

REPO_ROOT = Path(__file__).resolve().parent.parent
SKIP_NAMES = {"__pycache__", ".venv", ".git"}


def load_config(env_file: Path = REPO_ROOT / "pi.env") -> dict:
    config = {"PI_HOST": "", "PI_USER": "pi", "PI_PASSWORD": "", "PI_PROJECT_DIR": "~/cats-localization-v2"}
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                config[key.strip()] = value.strip()
    for key in config:
        config[key] = os.environ.get(key, config[key])
    if not config["PI_HOST"]:
        sys.exit(f"PI_HOST is not set: create {env_file} from pi.env.example")
    return config


class Pi:
    """A connection to the Pi with helpers to run commands and transfer files."""

    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        self.client = paramiko.SSHClient()
        # Local network only: accept the Pi's host key on first use.
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.client.connect(
            self.config["PI_HOST"],
            username=self.config["PI_USER"],
            password=self.config["PI_PASSWORD"],
            timeout=15,
            look_for_keys=False,
            allow_agent=False,
        )
        self._sftp = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        if self._sftp:
            self._sftp.close()
        self.client.close()

    @property
    def sftp(self) -> paramiko.SFTPClient:
        if self._sftp is None:
            self._sftp = self.client.open_sftp()
        return self._sftp

    def run(self, command: str, sudo: bool = False, echo: bool = True, timeout: float | None = None) -> tuple[int, str]:
        """Run a command, streaming its combined stdout/stderr. Returns (exit code, output)."""
        if sudo:
            command = f"sudo -S -p '' sh -c {shlex.quote(command)}"
        stdin, stdout, _ = self.client.exec_command(command, timeout=timeout)
        stdout.channel.set_combine_stderr(True)
        if sudo:
            stdin.write(self.config["PI_PASSWORD"] + "\n")
            stdin.flush()
        lines = []
        for line in iter(stdout.readline, ""):
            lines.append(line)
            if echo:
                print(line, end="", flush=True)
        return stdout.channel.recv_exit_status(), "".join(lines)

    def remote_path(self, path: str) -> str:
        """Expand a leading `~` (SFTP does not) to the remote home directory."""
        if path == "~" or path.startswith("~/"):
            return self.sftp.normalize(".") + path[1:]
        return path

    def makedirs(self, remote_dir: str):
        parts = []
        current = remote_dir
        while current not in ("", "/"):
            try:
                self.sftp.stat(current)
                break
            except FileNotFoundError:
                parts.append(current)
                current = posixpath.dirname(current)
        for directory in reversed(parts):
            self.sftp.mkdir(directory)

    def put(self, local: str | Path, remote: str):
        """Upload a file or a folder (recursively) to `remote`."""
        local, remote = Path(local), self.remote_path(remote)
        if local.is_dir():
            for root, dirs, files in os.walk(local):
                dirs[:] = [d for d in dirs if d not in SKIP_NAMES]
                rel = Path(root).relative_to(local).as_posix()
                target_dir = remote if rel == "." else posixpath.join(remote, rel)
                self.makedirs(target_dir)
                for name in files:
                    self.sftp.put(str(Path(root) / name), posixpath.join(target_dir, name))
        else:
            self.makedirs(posixpath.dirname(remote))
            self.sftp.put(str(local), remote)

    def get(self, remote: str, local: str | Path):
        """Download a file or a folder (recursively) from `remote`."""
        remote, local = self.remote_path(remote), Path(local)
        if stat.S_ISDIR(self.sftp.stat(remote).st_mode):
            local.mkdir(parents=True, exist_ok=True)
            for entry in self.sftp.listdir_attr(remote):
                if entry.filename not in SKIP_NAMES:
                    self.get(posixpath.join(remote, entry.filename), local / entry.filename)
        else:
            local.parent.mkdir(parents=True, exist_ok=True)
            self.sftp.get(remote, str(local))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    run_p = sub.add_parser("run", help="run a shell command on the Pi")
    run_p.add_argument("command")
    run_p.add_argument("--sudo", action="store_true", help="run with sudo (password from pi.env)")
    put_p = sub.add_parser("put", help="upload a file or folder")
    put_p.add_argument("local")
    put_p.add_argument("remote")
    get_p = sub.add_parser("get", help="download a file or folder")
    get_p.add_argument("remote")
    get_p.add_argument("local")
    args = parser.parse_args()

    with Pi() as pi:
        if args.action == "run":
            code, _ = pi.run(args.command, sudo=args.sudo)
            sys.exit(code)
        if args.action == "put":
            pi.put(args.local, args.remote)
        else:
            pi.get(args.remote, args.local)


if __name__ == "__main__":
    main()

#!/bin/bash
# Install the autostart service (cats-app.service) and the power-off permission (cats-poweroff.sudoers) on the Pi.
# Run from the project folder after scripts/deploy.py:
#     sudo bash pi/system/install.sh            # install, enable at boot and (re)start the app
#     sudo bash pi/system/install.sh --remove   # stop and disable the app, remove both files
set -euo pipefail
HERE=$(dirname "$(readlink -f "$0")")
UNIT=/etc/systemd/system/cats-app.service
SUDOERS=/etc/sudoers.d/cats-poweroff  # no dot in the name: sudo skips files in sudoers.d that contain one
LOG=/home/pi/cats-localization-v2/results/phase6/service.log

if [[ ${1:-} == --remove ]]; then
    systemctl disable --now cats-app 2>/dev/null || true
    rm -f "$UNIT" "$SUDOERS"
    systemctl daemon-reload
    echo "cats-app removed"
    exit 0
fi

visudo -cq -f "$HERE/cats-poweroff.sudoers"  # never install a sudoers file that does not parse
install -m 0440 -o root -g root "$HERE/cats-poweroff.sudoers" "$SUDOERS"
install -m 0644 -o root -g root "$HERE/cats-app.service" "$UNIT"
install -d -o pi -g pi "$(dirname "$LOG")"
[[ -e $LOG ]] || install -m 0644 -o pi -g pi /dev/null "$LOG"  # owned by pi, not created by systemd as root
systemctl daemon-reload
systemctl enable cats-app
systemctl restart cats-app
echo "cats-app enabled and started; log: $LOG"

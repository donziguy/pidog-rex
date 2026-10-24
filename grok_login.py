#!/usr/bin/env python3
"""Sign Rex into a Grok subscription (SuperGrok / X Premium+).

On the Pi, from the Rex checkout:

    python3 grok_login.py

A verification URL and a short code are printed. Open the URL on any device,
enter the code, and approve. The access token and refresh token are written
into ~/.config/pidog/security_dog.env (mode 600). Nothing else is printed:
not the device code, not the tokens.

Then reload the user service so it picks up the file:

    systemctl --user restart rex
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import grok_oauth


def main() -> int:
    return grok_oauth.run_device_login()


if __name__ == "__main__":
    sys.exit(main())

"""Zeek conn.log -> NIDS backend sender.

Run this on EACH authorized machine that runs Zeek (yours, your friend's...).
It only forwards flows that its own Zeek sensor sees.

Examples:
    python zeek_sender.py
    python zeek_sender.py --log /path/to/conn.log --device-id 3
    python zeek_sender.py --from-start        # also send old lines already in conn.log

Optional environment variables (skip the prompts): NIDS_EMAIL, NIDS_PASSWORD
"""
import argparse
import getpass
import os
import time
from pathlib import Path

import requests

DEFAULT_API = "https://aiml-multi-user-nids-1.onrender.com"
DEFAULT_LOG = Path(__file__).resolve().parent / "zeek-live" / "live" / "conn.log"

BATCH_SIZE = 10          # send when this many flows are waiting...
FLUSH_SECONDS = 5        # ...or when the oldest one has waited this long
MAX_PENDING = 5000       # if the API is down, keep at most this many flows
MAX_PER_REQUEST = 200


class Api:
    def __init__(self, base_url, email, password):
        self.base_url = base_url.rstrip("/")
        self.email = email
        self.password = password
        self.token = None

    def login(self):
        # Render free tier can take ~1 minute to wake up, so use a long timeout.
        response = requests.post(
            f"{self.base_url}/login",
            json={"email": self.email, "password": self.password},
            timeout=90,
        )
        response.raise_for_status()
        self.token = response.json()["access_token"]

    def request(self, method, path, **kwargs):
        """Authenticated request. Logs in again once if the token expired (401)."""
        kwargs.setdefault("timeout", 60)
        for attempt in (1, 2):
            if not self.token:
                self.login()
            headers = {"Authorization": f"Bearer {self.token}"}
            response = requests.request(
                method, f"{self.base_url}{path}", headers=headers, **kwargs
            )
            if response.status_code == 401 and attempt == 1:
                self.token = None
                continue
            response.raise_for_status()
            return response
        raise RuntimeError("Authentication failed")


def choose_device(api, forced_id=None):
    devices = api.request("GET", "/devices").json()
    active = [d for d in devices if d.get("is_active")]
    if not active:
        raise RuntimeError("No active devices on this account. Register one in the dashboard first.")

    print("\nRegistered devices:")
    for device in active:
        print(f'  {device["id"]}: {device["device_name"]}')

    if forced_id is not None:
        if not any(d["id"] == forced_id for d in active):
            raise ValueError(f"Device ID {forced_id} is not one of your active devices.")
        return forced_id
    if len(active) == 1:
        return active[0]["id"]

    selected = int(input("Enter the Device ID of THIS machine: "))
    if not any(d["id"] == selected for d in active):
        raise ValueError("Invalid device ID.")
    return selected


def clean(value):
    return None if value in (None, "", "-") else value


def to_int(value):
    value = clean(value)
    return int(value) if value is not None else None


def to_float(value):
    value = clean(value)
    return float(value) if value is not None else None


def convert_record(record):
    return {
        "uid": clean(record.get("uid")),
        "timestamp": float(record["ts"]),
        "source_ip": clean(record.get("id.orig_h")),
        "source_port": to_int(record.get("id.orig_p")),
        "destination_ip": clean(record.get("id.resp_h")),
        "destination_port": to_int(record.get("id.resp_p")),
        "protocol": clean(record.get("proto")),
        "service": clean(record.get("service")),
        "connection_state": clean(record.get("conn_state")),
        "duration": to_float(record.get("duration")),
        "orig_bytes": to_int(record.get("orig_bytes")),
        "resp_bytes": to_int(record.get("resp_bytes")),
        "orig_pkts": to_int(record.get("orig_pkts")),
        "resp_pkts": to_int(record.get("resp_pkts")),
    }


def read_fields(path):
    """Return the column names from the '#fields' header of a Zeek log."""
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("#fields"):
                return line.rstrip("\r\n").split("\t")[1:]
            if not line.startswith("#"):
                break
    return None


def send_pending(api, device_id, pending):
    """Send pending flows. Returns how many were saved; raises on API failure."""
    sent = 0
    while pending:
        chunk = pending[:MAX_PER_REQUEST]
        response = api.request(
            "POST", "/sensor/ingest", json={"device_id": device_id, "flows": chunk}
        )
        saved = response.json().get("records_saved", len(chunk))
        del pending[: len(chunk)]
        sent += saved
    return sent


def follow(api, device_id, log_file, from_start):
    pending = []
    last_flush = time.time()
    backoff = 3
    handle = None
    fields = None
    start_at_beginning = from_start

    print(f"Watching Zeek log: {log_file}")
    print("Sending only NEW flows." if not from_start else "Sending the whole existing log too.")

    while True:
        # Wait for Zeek to create the log
        if not log_file.exists():
            print("Waiting for Zeek conn.log ...")
            time.sleep(3)
            continue

        # (Re)open the log
        if handle is None:
            fields = read_fields(log_file)
            if not fields:
                print("conn.log has no '#fields' header yet; waiting ...")
                time.sleep(2)
                continue
            handle = open(log_file, "r", encoding="utf-8", errors="replace")
            if start_at_beginning:
                handle.seek(0)
            else:
                handle.seek(0, os.SEEK_END)
            start_at_beginning = False

        # Zeek restarted and truncated/replaced the file -> reopen from the top
        if log_file.stat().st_size < handle.tell():
            print("conn.log was reset; reopening.")
            handle.close()
            handle = None
            start_at_beginning = True
            continue

        position = handle.tell()
        line = handle.readline()

        if line and not line.endswith("\n"):
            # Zeek is still writing this line; try again shortly
            handle.seek(position)
            line = ""

        if line:
            line = line.rstrip("\r\n")
            if line.startswith("#fields"):
                fields = line.split("\t")[1:]
            elif line and not line.startswith("#"):
                values = line.split("\t")
                if len(values) == len(fields):
                    try:
                        pending.append(convert_record(dict(zip(fields, values))))
                    except (ValueError, KeyError) as error:
                        print("Skipping bad Zeek record:", error)
            if len(pending) < BATCH_SIZE:
                continue
        else:
            time.sleep(0.5)

        # Flush when the batch is full or has waited long enough
        due = len(pending) >= BATCH_SIZE or (
            pending and time.time() - last_flush >= FLUSH_SECONDS
        )
        if not due:
            continue

        try:
            saved = send_pending(api, device_id, pending)
            print(f"Saved: {saved}")
            backoff = 3
        except requests.RequestException as error:
            print(f"API error: {error} (will retry in {backoff}s, {len(pending)} flows waiting)")
            if len(pending) > MAX_PENDING:
                del pending[: len(pending) - MAX_PENDING]
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)
        last_flush = time.time()


def main():
    parser = argparse.ArgumentParser(description="Send Zeek conn.log flows to the NIDS backend")
    parser.add_argument("--api", default=os.getenv("NIDS_API_URL", DEFAULT_API))
    parser.add_argument("--log", default=os.getenv("ZEEK_CONN_LOG", str(DEFAULT_LOG)))
    parser.add_argument("--device-id", type=int, default=None)
    parser.add_argument("--from-start", action="store_true",
                        help="also send lines already in conn.log (default: only new lines)")
    args = parser.parse_args()

    email = os.getenv("NIDS_EMAIL") or input("Enter your registered email: ").strip()
    password = os.getenv("NIDS_PASSWORD") or getpass.getpass("Account password: ")

    api = Api(args.api, email, password)
    api.login()
    print("Login successful!")

    device_id = choose_device(api, args.device_id)
    print("Using Device ID:", device_id)

    try:
        follow(api, device_id, Path(args.log), args.from_start)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()

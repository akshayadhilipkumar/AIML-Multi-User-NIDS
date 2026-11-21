import csv
import getpass
import time
import requests
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
LOG_FILE = PROJECT_DIR / "zeek-live" / "live" / "conn.log"

API_URL = "https://aiml-multi-user-nids-1.onrender.com"
EMAIL = input("Enter your registered email: ").strip()
BATCH_SIZE = 10


def login():
    password = getpass.getpass("Account password: ")
    response = requests.post(
        f"{API_URL}/login",
        json={"email": EMAIL, "password": password},
        timeout=60
    )
    
    response.raise_for_status()
    print("Login successful!")
    return response.json()["access_token"]

def get_device_id(token):
    response = requests.get(
        f"{API_URL}/devices",
        headers={"Authorization": f"Bearer {token}"},
        timeout=10
    )
    response.raise_for_status()

    devices = response.json()

    if not devices:
        raise RuntimeError("No registered devices found.")

    active_devices = [d for d in devices if d["is_active"]]

    if not active_devices:
        raise RuntimeError("No active devices found.")

    print("\nRegistered devices:")
    for device in active_devices:
        print(f'{device["id"]}: {device["device_name"]}')

    if len(active_devices) == 1:
        return active_devices[0]["id"]

    selected_id = int(input("Enter your Device ID: "))
    if not any(d["id"] == selected_id for d in active_devices):
        raise ValueError("Invalid device ID.")

    return selected_id

def convert_value(value):
    return None if value in (None, "", "-") else value


def to_int(value):
    value = convert_value(value)
    return int(value) if value is not None else None


def to_float(value):
    value = convert_value(value)
    return float(value) if value is not None else None


def convert_record(record):
    return {
        "uid": convert_value(record.get("uid")),
        "timestamp": float(record["ts"]),
        "source_ip": convert_value(record.get("id.orig_h")),
        "source_port": to_int(record.get("id.orig_p")),
        "destination_ip": convert_value(record.get("id.resp_h")),
        "destination_port": to_int(record.get("id.resp_p")),
        "protocol": convert_value(record.get("proto")),
        "service": convert_value(record.get("service")),
        "connection_state": convert_value(record.get("conn_state")),
        "duration": to_float(record.get("duration")),
        "orig_bytes": to_int(record.get("orig_bytes")),
        "resp_bytes": to_int(record.get("resp_bytes")),
        "orig_pkts": to_int(record.get("orig_pkts")),
        "resp_pkts": to_int(record.get("resp_pkts"))
    }


def send_batch(token, device_id, batch):
    response = requests.post(
        f"{API_URL}/sensor/ingest",
        headers={"Authorization": f"Bearer {token}"},
        json={"device_id": device_id, "flows": batch},
        timeout=20
    )
    response.raise_for_status()
    print("Saved:", response.json()["records_saved"])


def main():
    if not LOG_FILE.exists():
        print("Waiting for Zeek conn.log:", LOG_FILE)
        return

    token = login()
    device_id = get_device_id(token)
    print("Using Device ID:", device_id)
    print("Watching Zeek log:", LOG_FILE)

    with open(LOG_FILE, "r", encoding="utf-8") as file:
        fields = None
        batch = []

        while True:
            line = file.readline()

            if not line:
                if batch:
                    try:
                        send_batch(token, device_id, batch)
                        batch.clear()
                    except requests.RequestException as error:
                        print("API error:", error)
                time.sleep(1)
                continue

            line = line.rstrip("\r\n")

            if line.startswith("#separator"):
                continue

            if line.startswith("#fields"):
                fields = line.split("\t")[1:]
                continue

            if line.startswith("#") or not line or not fields:
                continue

            values = line.split("\t")

            if len(values) != len(fields):
                continue

            try:
                record = dict(zip(fields, values))
                batch.append(convert_record(record))

                if len(batch) >= BATCH_SIZE:
                    send_batch(token, device_id, batch)
                    batch.clear()

            except (ValueError, KeyError) as error:
                print("Invalid Zeek record:", error)
            except requests.RequestException as error:
                print("API error:", error)
                time.sleep(3)


if __name__ == "__main__":
    main()

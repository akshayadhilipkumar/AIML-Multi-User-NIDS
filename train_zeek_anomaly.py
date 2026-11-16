"""Train a prototype anomaly detector from Zeek conn.log flow metrics.

This is a separate model from the UNSW-NB15 CSV models. It learns the local
baseline as assumed-normal traffic and reports unusual flows, not confirmed
attacks. It does not overwrite any existing model files.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


INPUT_FILE = Path.home() / "Downloads" / "baseline_more_flows.tsv"
OUTPUT_FILE = Path.cwd() / "zeek_anomaly_model.pkl"

# This order matches the numeric TSV created with zeek-cut.
ALL_COLUMNS = [
    "duration",
    "orig_pkts",
    "resp_pkts",
    "orig_bytes",
    "resp_bytes",
    "orig_ip_bytes",
    "resp_ip_bytes",
]

# Use Zeek's application-payload byte counts as candidate traffic features.
# IP-layer byte totals are excluded here to avoid duplicating byte-volume data.
FEATURES = ["duration", "orig_pkts", "resp_pkts", "orig_bytes", "resp_bytes"]


def main():
    if not INPUT_FILE.is_file():
        raise FileNotFoundError(
            f"Could not find {INPUT_FILE}. First export baseline_more_flows.tsv from Zeek."
        )

    frame = pd.read_csv(
        INPUT_FILE,
        sep="\t",
        header=None,
        names=ALL_COLUMNS,
        na_values=["-"],
    )
    for column in ALL_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    features = frame[FEATURES].replace([np.inf, -np.inf], np.nan)
    features = features.dropna(how="all")
    if len(features) < 100:
        raise ValueError(
            f"Only {len(features)} usable flows found. Capture more ordinary traffic before training."
        )
    if (features.fillna(0) < 0).any().any():
        raise ValueError("Negative flow metrics found; inspect the Zeek export before training.")

    train_rows, held_out_rows = train_test_split(
        features, test_size=0.20, random_state=42
    )
    detector = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("log1p", FunctionTransformer(np.log1p, validate=False)),
            (
                "isolation_forest",
                IsolationForest(
                    n_estimators=300,
                    contamination="auto",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    detector.fit(train_rows)
    held_out_predictions = detector.predict(held_out_rows)
    flagged = int((held_out_predictions == -1).sum())

    joblib.dump(
        {
            "pipeline": detector,
            "features": FEATURES,
            "training_rows": len(train_rows),
            "held_out_rows": len(held_out_rows),
            "assumption": "Captured baseline traffic is assumed to be mostly normal.",
            "interpretation": "-1 means unusual flow; it does not confirm an attack.",
        },
        OUTPUT_FILE,
    )

    print(f"Usable baseline flows: {len(features):,}")
    print(f"Model training flows: {len(train_rows):,}")
    print(f"Held-out baseline flows: {len(held_out_rows):,}")
    print(f"Held-out flows flagged as unusual: {flagged:,}")
    print(f"Saved separate Zeek anomaly model to: {OUTPUT_FILE}")
    print("This check is not attack accuracy or proof of attack detection.")


if __name__ == "__main__":
    main()

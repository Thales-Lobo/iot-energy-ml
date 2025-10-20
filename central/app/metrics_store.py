# =================== central/app/metrics_store.py ===================
from __future__ import annotations

import os
import csv
import time
import threading
import logging
from typing import Dict, Any, List

import pandas as pd
from fastapi.responses import FileResponse

logger = logging.getLogger("central.metrics_store")


class MetricsStore:
    REQUIRED_FIELDS = [
        "node_id",
        "sent_ts",
        "received_at",
        "pred_label",
        "prob_attack",
        "is_attack",
        "correct",
    ]

    def __init__(self, csv_path: str = "data/results.csv") -> None:
        self.csv_path = csv_path
        self.records: List[Dict[str, Any]] = []
        self.lock = threading.Lock()
        self.start_time = time.time()

        # Ensure directory exists
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        # Create CSV with headers if it doesn't exist
        if not os.path.exists(csv_path):
            try:
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=self.REQUIRED_FIELDS)
                    writer.writeheader()
                logger.info(f"Created new CSV with headers at {csv_path}")
            except Exception:
                logger.exception("Failed to create CSV file at startup")

        logger.info(f"MetricsStore initialized, output -> {csv_path}")

    # ------------------------------------------------------------------
    def record(self, record: Dict[str, Any]) -> None:
        """Record a new inference result, only if valid, and flush immediately."""
        # --- Validate required fields ---
        for key in self.REQUIRED_FIELDS:
            if key not in record or record[key] is None:
                logger.warning("Skipping record: missing or invalid field '%s'", key)
                return

        # --- Validate content ---
        if record["sent_ts"] is None:
            logger.warning("Skipping record: sent_ts is None")
            return

        with self.lock:
            self.records.append(record)
            # --- Flush immediately ---
            self._flush_single(record)
            logger.debug("Recorded and flushed metrics for node=%s", record["node_id"])

    # ------------------------------------------------------------------
    def _flush_single(self, record: Dict[str, Any]) -> None:
        """Write a single record to CSV (append mode)"""
        try:
            file_exists = os.path.exists(self.csv_path)
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.REQUIRED_FIELDS)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(record)
            # Clear in-memory buffer of that record
            self.records.remove(record)
        except Exception:
            logger.exception("Failed to flush single record to CSV")

    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        with self.lock:
            # Load all records from disk to compute summary
            try:
                df = pd.read_csv(self.csv_path)
            except FileNotFoundError:
                df = pd.DataFrame(columns=self.REQUIRED_FIELDS)

            total = len(df)
            correct = df["correct"].sum() if total > 0 else 0
            accuracy = round(correct / total, 4) if total > 0 else None

            # Accuracy by 60-second bucket
            bucket_size = 60
            bucket_totals = {}
            bucket_corrects = {}
            for _, r in df.iterrows():
                ts = r.get("sent_ts", None)
                if pd.isna(ts):
                    continue
                bucket = int(float(ts) // bucket_size)
                bucket_totals[bucket] = bucket_totals.get(bucket, 0) + 1
                if int(r["correct"]) == 1:
                    bucket_corrects[bucket] = bucket_corrects.get(bucket, 0) + 1

            acc_by_bucket = {
                b: round(bucket_corrects.get(b, 0) / t, 3)
                for b, t in bucket_totals.items()
            }

            return {
                "total": total,
                "correct": int(correct),
                "accuracy": accuracy,
                "accuracy_by_time_bucket": acc_by_bucket,
                "uptime_sec": round(time.time() - self.start_time, 2),
            }

    # ------------------------------------------------------------------
    def flush(self) -> None:
        """Flush all in-memory records (if any remain) to CSV"""
        with self.lock:
            if not self.records:
                return
            for r in self.records.copy():
                self._flush_single(r)

    # ------------------------------------------------------------------
    def load(self) -> None:
        """Load existing CSV records into memory (if needed)"""
        if not os.path.exists(self.csv_path):
            logger.info("No existing CSV to load; starting fresh.")
            return
        try:
            df = pd.read_csv(self.csv_path)
            valid_records = [
                {k: v for k, v in row.items() if k in self.REQUIRED_FIELDS}
                for row in df.to_dict(orient="records")
            ]
            with self.lock:
                self.records.extend(valid_records)
            logger.info(f"Loaded {len(valid_records)} historical records from {self.csv_path}")
        except Exception:
            logger.exception("Failed to load existing CSV; continuing empty")

    # ------------------------------------------------------------------
    def reset(self) -> None:
        with self.lock:
            self.records.clear()
        logger.info("MetricsStore reset: memory cleared")

    # ------------------------------------------------------------------
    def get_csv_response(self) -> FileResponse:
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")
        return FileResponse(self.csv_path, media_type="text/csv", filename=os.path.basename(self.csv_path))

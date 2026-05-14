"""
GSM Tool — Plotly Dashboard v2 | Data Loader

Reads 5 CSV files from the data/ folder and produces a single combined
DataFrame ready for the dashboard to use.

This replaces the live Oracle database queries with equivalent pandas operations.

Expected files in plotly_dashboard_v2/data/:
  cgci.csv       -- Customer / sales hierarchy (one row per customer entity)
  cpdb_fxg.csv   -- FXG monthly performance data (one row per customer per month)
  cpdb_fxed.csv  -- FXED monthly performance data (same columns as FXG)
  opdays.csv     -- US operating days per month (OPDAYS5 = 5-day week)
  bp_goals.csv   -- Annual BP growth goal rates per OPCO + segment

Tip: copy the data/ folder from plotly_dashboard_v1/ — the CSV files are identical.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# All CSV files must be in a folder called "data" next to this file
DATA_DIR = Path(__file__).parent / "data"

# These are the numeric columns in the performance CSVs.
# We'll force them to numbers in case they were read as strings.
NUMERIC_PERFORMANCE_COLUMNS = [
    "NET_REV_AMT",          # revenue including fuel surcharge
    "NET_REV_AMT_WOF",      # revenue excluding fuel surcharge
    "FUEL_SRCHG_AMT",       # fuel surcharge amount
    "NET_REV_AMT_MIN_RBTE", # revenue minus rebates
    "SHP_DISC_AMT",         # shipping discounts
    "SHP_QTY",              # shipment count
    "SHP_RATE_WGT",         # rated weight
    "SHP_VOL_QTY",          # volume (main volume metric used in the dashboard)
    "TTL_SRCHG_AMT",        # total surcharges with fuel
    "TTL_SRCHG_AMT_WOF",    # total surcharges without fuel
    "NET_REV_AMT_CMPTD_WF", # computed revenue with fuel
    "OPDAYS5",              # 5-day operating days for this month
    "OPDAYS7",              # 7-day operating days for this month
]

NUMERIC_GOALS_COLUMNS = [
    "REV_GOAL_WF",   # revenue growth goal rate (fuel included)
    "REV_GOAL_WOF",  # revenue growth goal rate (fuel excluded)
    "VOL_GOAL",      # volume growth goal rate
    "YLD_GOAL_WF",   # yield growth goal rate (fuel included)
    "YLD_GOAL_WOF",  # yield growth goal rate (fuel excluded)
]


def load_performance_data():
    """
    Loads and joins all performance CSV files into one combined DataFrame.

    Steps:
      1. Read FXG data and FXED data, then stack them (UNION)
      2. Join customer/sales hierarchy from cgci (LEFT JOIN on CTRY_ENTI_NBR)
      3. Join operating days from opdays (LEFT JOIN on SHIPMNTH date)
      4. Force numeric columns to be numbers (not strings)
      5. Add helper columns: Year, MonthNum, YM (for easy filtering later)

    Returns a pandas DataFrame with one row per customer per month.
    """

    # ── Step 1: Read all four source files ────────────────────────────────────
    cgci   = pd.read_csv(DATA_DIR / "cgci.csv")        # customer hierarchy
    fxg    = pd.read_csv(DATA_DIR / "cpdb_fxg.csv")    # FXG performance rows
    fxed   = pd.read_csv(DATA_DIR / "cpdb_fxed.csv")   # FXED performance rows
    opdays = pd.read_csv(DATA_DIR / "opdays.csv")       # operating days

    # ── Step 2: Stack FXG + FXED into one table (they have the same columns) ──
    # This is the equivalent of SQL UNION ALL.
    combined_perf = pd.concat([fxg, fxed], ignore_index=True)

    # ── Step 3: Parse the SHIPMNTH column into proper Python dates ────────────
    # The Oracle export uses "01-JUN-2024" format; ISO format "2024-06-01" also works.
    combined_perf["SHIPMNTH"] = pd.to_datetime(
        combined_perf["SHIPMNTH"], dayfirst=True, errors="coerce"
    )
    opdays["SHIPMNTH"] = pd.to_datetime(
        opdays["SHIPMNTH"], dayfirst=True, errors="coerce"
    )

    # ── Step 4: Decide which columns to use for the hierarchy join ────────────
    # If both tables have GLBL_ENTI_NBR, include it to make the join more precise.
    if "GLBL_ENTI_NBR" in combined_perf.columns and "GLBL_ENTI_NBR" in cgci.columns:
        join_keys = ["CTRY_ENTI_NBR", "GLBL_ENTI_NBR"]
    else:
        join_keys = ["CTRY_ENTI_NBR"]

    # Only bring these columns over from cgci (the rest are already in performance)
    hierarchy_columns = [
        "SLS_SVP_EMP_NM",        # Senior VP
        "SLS_VP_EMP_NM",         # VP
        "SLS_DIR_MGR_NM",        # Director / Manager
        "PRCG_DIR_MGR_NM",       # Pricing Director
        "SEG_ROLLUP_DESC",       # Segment rollup (e.g. LARGE, SAM)
        "SEG_GRP_DESC",          # Segment group
        "GLBL_SEG_DESC",         # Global segment description
    ]
    columns_to_keep = join_keys + [c for c in hierarchy_columns if c in cgci.columns]

    # Drop duplicate cgci rows for the same join key so we don't multiply rows
    cgci_deduped = cgci[columns_to_keep].drop_duplicates(subset=join_keys)

    # ── Step 5: LEFT JOIN performance + hierarchy ─────────────────────────────
    # Every performance row gets enriched with SVP / VP / DIR / segment info.
    combined = combined_perf.merge(cgci_deduped, on=join_keys, how="left")

    # ── Step 6: LEFT JOIN with operating days ─────────────────────────────────
    # Brings in OPDAYS5 (and OPDAYS7 if present) matched by month.
    opday_columns = ["SHIPMNTH"] + [
        col for col in ("OPDAYS5", "OPDAYS7") if col in opdays.columns
    ]
    combined = combined.merge(opdays[opday_columns], on="SHIPMNTH", how="left")

    # ── Step 7: Force all metric columns to numeric ───────────────────────────
    # Handles cases where CSV values were read as strings (e.g. "1,234" with commas).
    for col in NUMERIC_PERFORMANCE_COLUMNS:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce").fillna(0)

    # ── Step 8: Add date helper columns used throughout the dashboard ─────────
    combined["Year"]     = combined["SHIPMNTH"].dt.year       # e.g. 2025
    combined["MonthNum"] = combined["SHIPMNTH"].dt.month      # e.g. 6 for June
    combined["YM"]       = combined["SHIPMNTH"].dt.strftime("%Y-%m")  # e.g. "2025-06"

    return combined


def load_goals_data():
    """
    Reads bp_goals.csv which contains annual BP growth goal rates.

    If the file doesn't exist, returns an empty DataFrame with the correct columns
    so the dashboard still runs — it will just show a warning where goals are used.

    Returns a DataFrame with columns:
      OPCO_BREAKOUT, SEG_ROLLUP_DESC, SEG_GRP_DESC,
      REV_GOAL_WF, REV_GOAL_WOF, VOL_GOAL, YLD_GOAL_WF, YLD_GOAL_WOF
    """
    goals_file_path = DATA_DIR / "bp_goals.csv"

    # Gracefully handle missing file — return empty frame instead of crashing
    if not goals_file_path.exists():
        return pd.DataFrame(
            columns=["OPCO_BREAKOUT", "SEG_ROLLUP_DESC", "SEG_GRP_DESC"]
                    + NUMERIC_GOALS_COLUMNS
        )

    goals_df = pd.read_csv(goals_file_path)

    # Force goal rate columns to numeric
    for col in NUMERIC_GOALS_COLUMNS:
        if col in goals_df.columns:
            goals_df[col] = pd.to_numeric(goals_df[col], errors="coerce")

    return goals_df

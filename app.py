"""
GSM Tool — Plotly Dashboard v2
Simplified version: Visuals 01, 02, and 03 only. No Charts tab.
Beginner-friendly code with descriptive names and step-by-step comments.

Run:  python app.py   (from plotly_dashboard_v2/ directory)
      Opens at http://127.0.0.1:8051

Visual 01 — Previous Year monthly summary (FY25: Jun 2024 – May 2025)
Visual 02 — Current Year YTD with YoY% and normalisation (FY26)
Visual 03 — Full FY26: actuals for completed months + projections for the rest
"""

import dash
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np

from data_loader import load_performance_data

# ============================================================================
# START THE APP & LOAD DATA
# ============================================================================

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server  # needed if you deploy with gunicorn or waitress

# Load all CSV data once at startup (not on every user interaction)
all_data = load_performance_data()

# All 12 months in FY26 (Jun 2025 – May 2026) as "YYYY-MM" strings.
# Visual 03 iterates over this list to show every month, even future ones.
FY26_MONTHS = [
    "2025-06", "2025-07", "2025-08", "2025-09", "2025-10", "2025-11",
    "2025-12", "2026-01", "2026-02", "2026-03", "2026-04", "2026-05",
]

# Unique values for the sidebar dropdowns (computed once from the loaded data)
all_opcos = sorted(all_data["OPCO_BREAKOUT"].dropna().unique())
all_svps  = sorted(all_data["SLS_SVP_EMP_NM"].dropna().unique())

# Colours used in YoY% cells
COLOUR_POSITIVE = "#2ca02c"   # green — revenue / volume is growing
COLOUR_NEGATIVE = "#d62728"   # red   — revenue / volume is declining


# ============================================================================
# HELPER 1 — GET THE MATCHING PRIOR-YEAR MONTH
# ============================================================================

def get_prior_year_month(year_month):
    """
    Given a month string like "2025-07", return the same month one year earlier: "2024-07".

    Why: We use this to find the FY25 month that corresponds to each FY26 month,
    so we can calculate YoY% and operating-day normalization.

    Example:
      get_prior_year_month("2025-11") → "2024-11"
      get_prior_year_month("2026-01") → "2025-01"
    """
    year_str, month_str = year_month.split("-")
    prior_year = int(year_str) - 1
    return f"{prior_year}-{month_str}"


# ============================================================================
# HELPER 2 — AGGREGATE MANY ROWS INTO ONE ROW PER MONTH
# ============================================================================

def aggregate_by_month(data, revenue_column):
    """
    Takes a DataFrame with many rows (one per customer per month) and collapses
    it into one row per month with summed Revenue and Volume.

    Also calculates:
      Yield  = Revenue ÷ Volume  (revenue per unit shipped)
      ADV    = Volume ÷ OpDays   (average daily volume)

    Parameters
    ----------
    data           : DataFrame for one fiscal year (already filtered)
    revenue_column : "NET_REV_AMT" or "NET_REV_AMT_WOF" depending on fuel toggle

    Returns a DataFrame with one row per month, sorted oldest → newest.
    """
    monthly_summary = data.groupby("YM").agg(
        Revenue=(revenue_column, "sum"),     # total revenue this month
        Volume=("SHP_VOL_QTY",  "sum"),     # total volume this month
        OpDays=("OPDAYS5",      "max"),      # operating days (same for every customer in a month)
    ).reset_index()

    # Yield: how much revenue per unit shipped
    monthly_summary["Yield"] = (
        monthly_summary["Revenue"] / monthly_summary["Volume"].replace(0, np.nan)
    )

    # ADV: how many units shipped per working day on average
    monthly_summary["ADV"] = (
        monthly_summary["Volume"] / monthly_summary["OpDays"].replace(0, np.nan)
    )

    monthly_summary = monthly_summary.sort_values("YM").reset_index(drop=True)
    return monthly_summary


# ============================================================================
# HELPER 3 — ATTACH PRIOR-YEAR VALUES AND NORMALISATION TO THE CY TABLE
# ============================================================================

def add_prior_year_columns(cy_monthly, py_monthly):
    """
    For each CY (FY26) month, look up the matching FY25 month and attach its
    Revenue, Volume, Yield, and OpDays as new columns (PY_Rev, PY_Vol, etc.).

    Then computes normalised values to make fair month-to-month comparisons:
      NormFactor   = PY_OpDays ÷ CY_OpDays
      NormRevenue  = CY_Revenue × NormFactor

    If November 2025 has 19 operating days but November 2024 had 20, then
    NormRevenue scales CY revenue up so it reflects what we would have earned
    with 20 days — making the YoY comparison fair.

    Also computes YoY% columns:
      RevYoY   = (CY_Revenue ÷ PY_Revenue − 1) × 100
      VolYoY   = (CY_Volume  ÷ PY_Volume  − 1) × 100
      YieldYoY = (CY_Yield   ÷ PY_Yield   − 1) × 100

    Returns an enriched copy of cy_monthly.
    """
    cy_enriched = cy_monthly.copy()

    # Build a dictionary for fast lookup: "2024-07" → {"OpDays": 23, "Revenue": ..., ...}
    py_lookup = py_monthly.set_index("YM")

    # Loop through each CY month and attach the matching PY values
    for row_index, row in cy_enriched.iterrows():
        prior_ym = get_prior_year_month(row["YM"])  # e.g. "2025-08" → "2024-08"

        cy_enriched.at[row_index, "PY_OpDays"] = py_lookup["OpDays"].get(prior_ym, np.nan)
        cy_enriched.at[row_index, "PY_Rev"]    = py_lookup["Revenue"].get(prior_ym, np.nan)
        cy_enriched.at[row_index, "PY_Vol"]    = py_lookup["Volume"].get(prior_ym, np.nan)
        cy_enriched.at[row_index, "PY_Yield"]  = py_lookup["Yield"].get(prior_ym, np.nan)

    # NormFactor: ratio of PY opdays to CY opdays for each month
    cy_enriched["NormFactor"] = (
        cy_enriched["PY_OpDays"] / cy_enriched["OpDays"].replace(0, np.nan)
    )

    # Normalised CY metrics — adjusted to the same opday count as PY
    cy_enriched["NormRevenue"] = cy_enriched["Revenue"] * cy_enriched["NormFactor"]
    cy_enriched["NormVolume"]  = cy_enriched["Volume"]  * cy_enriched["NormFactor"]
    cy_enriched["NormYield"]   = (
        cy_enriched["NormRevenue"] / cy_enriched["NormVolume"].replace(0, np.nan)
    )

    # Year-over-Year percentage changes
    cy_enriched["RevYoY"] = (
        cy_enriched["Revenue"] / cy_enriched["PY_Rev"].replace(0, np.nan) - 1
    ) * 100
    cy_enriched["VolYoY"] = (
        cy_enriched["Volume"] / cy_enriched["PY_Vol"].replace(0, np.nan) - 1
    ) * 100
    cy_enriched["YieldYoY"] = (
        cy_enriched["Yield"] / cy_enriched["PY_Yield"].replace(0, np.nan) - 1
    ) * 100

    return cy_enriched


# ============================================================================
# HELPER 4 — PROJECT FUTURE MONTHS
# ============================================================================

def calculate_projections(cy_monthly, py_monthly):
    """
    For FY26 months that don't have actuals yet, estimate revenue and volume
    based on the pace CY is running at compared to PY.

    Algorithm (from projections_poc_v2.py):
      1. Sum all actual CY revenue/volume, normalized for operating-day differences.
         → cy_norm_revenue_total, cy_norm_volume_total
      2. Sum the matching PY months' revenue/volume (same months as #1).
         → py_attainment_revenue, py_attainment_volume
      3. Compute pace ratios:
         revenue_pace_ratio = cy_norm_revenue_total / py_attainment_revenue
         volume_pace_ratio  = cy_norm_volume_total  / py_attainment_volume
      4. For each future month M:
         projected_revenue = revenue_pace_ratio × PY_revenue_for_M
         projected_volume  = volume_pace_ratio  × PY_volume_for_M

    Returns (projections_df, last_actual_month) where:
      projections_df    has columns: YM, Proj_Rev, Proj_Vol, Proj_Yld
      last_actual_month is the latest "YYYY-MM" string with real CY data
    """
    py_lookup = py_monthly.set_index("YM")

    # Only consider months that actually have CY data (Revenue > 0)
    cy_actuals = cy_monthly[cy_monthly["Revenue"] > 0].copy()
    if cy_actuals.empty:
        return pd.DataFrame(), None

    last_actual_month = cy_actuals["YM"].max()
    last_py_month     = get_prior_year_month(last_actual_month)

    # ── Step A: Compute normalised CY totals across all actual months ─────────
    cy_norm_revenue_total = 0.0
    cy_norm_volume_total  = 0.0

    for _, row in cy_actuals.iterrows():
        prior_ym = get_prior_year_month(row["YM"])

        if prior_ym in py_lookup.index:
            py_opdays = py_lookup.at[prior_ym, "OpDays"]
        else:
            py_opdays = row["OpDays"]  # fall back to CY opdays if PY data is missing

        if row["OpDays"] > 0:
            norm_factor = py_opdays / row["OpDays"]
        else:
            norm_factor = 1.0

        cy_norm_revenue_total += row["Revenue"] * norm_factor
        cy_norm_volume_total  += row["Volume"]  * norm_factor

    # ── Step B: Sum PY revenue/volume over the same months CY has actuals for ─
    py_attainment_revenue = py_monthly[py_monthly["YM"] <= last_py_month]["Revenue"].sum()
    py_attainment_volume  = py_monthly[py_monthly["YM"] <= last_py_month]["Volume"].sum()

    if py_attainment_revenue == 0 or py_attainment_volume == 0:
        return pd.DataFrame(), last_actual_month

    # ── Step C: Project each future month ─────────────────────────────────────
    projection_rows = []

    for future_month in FY26_MONTHS:
        if future_month <= last_actual_month:
            continue  # skip months that already have actuals

        prior_ym = get_prior_year_month(future_month)
        if prior_ym not in py_lookup.index:
            continue  # skip if no PY data exists for this month

        projected_revenue = round(
            cy_norm_revenue_total / py_attainment_revenue * py_lookup.at[prior_ym, "Revenue"], 2
        )
        projected_volume = round(
            cy_norm_volume_total / py_attainment_volume * py_lookup.at[prior_ym, "Volume"], 2
        )
        projected_yield = projected_revenue / projected_volume if projected_volume else np.nan

        projection_rows.append({
            "YM":       future_month,
            "Proj_Rev": projected_revenue,
            "Proj_Vol": projected_volume,
            "Proj_Yld": projected_yield,
        })

    projections_df = pd.DataFrame(projection_rows)
    return projections_df, last_actual_month


# ============================================================================
# HELPER 5 — FORMAT A NUMBER FOR DISPLAY IN THE TABLE
# ============================================================================

def format_value(value, format_type):
    """
    Converts a raw number into a nicely formatted string for table cells.

    format_type   example output
    -----------   --------------
    "rev"       → "$1,234,567"
    "vol"       → "1,234,567"
    "yld"       → "$1.2345"
    "adv"       → "1,234.5"
    "pct"       → "3.45%"
    "pct+"      → "+3.45%" or "-1.23%"   (used for YoY — always shows the sign)
    "rev+"      → "+$1,234" or "-$567"
    "yld+"      → "+$0.1234"
    """
    if pd.isna(value):
        return ""    # show nothing for missing / NaN values
    if format_type == "rev":  return f"${value:,.0f}"
    if format_type == "vol":  return f"{value:,.0f}"
    if format_type == "yld":  return f"${value:.4f}"
    if format_type == "adv":  return f"{value:,.1f}"
    if format_type == "pct":  return f"{value:.2f}%"
    if format_type == "pct+": return f"{value:+.2f}%"
    if format_type == "rev+": return f"${value:+,.0f}"
    if format_type == "yld+": return f"${value:+.4f}"
    return str(value)


# ============================================================================
# HELPER 6 — MAKE A DASH DATATABLE WITH CONSISTENT STYLING
# ============================================================================

def make_table(rows, column_definitions, conditional_styles=None):
    """
    Wraps Dash's DataTable with a consistent look used across all three visuals.

    Parameters
    ----------
    rows               : list of dicts — each dict is one table row,
                         keys are column IDs, values are already-formatted strings
    column_definitions : list of {"name": "Display Name", "id": "col_id"} dicts
    conditional_styles : optional list of style rules for colour-coding, bold rows, etc.
    """
    return dash_table.DataTable(
        data=rows,
        columns=column_definitions,
        style_table={"overflowX": "auto"},
        style_header={
            "fontWeight": "bold",
            "backgroundColor": "#f0f2f6",
            "border": "1px solid #dee2e6",
            "textAlign": "center",
            "fontSize": "12px",
        },
        style_cell={
            "textAlign": "center",
            "padding": "5px 8px",
            "border": "1px solid #dee2e6",
            "fontFamily": "monospace",
            "fontSize": "12px",
        },
        style_data={"backgroundColor": "white"},
        style_data_conditional=conditional_styles or [],
        page_action="none",
    )


def column_defs(column_names):
    """
    Converts a plain list of column names into the format Dash DataTable expects.
    Example: ["Month", "Revenue"] → [{"name": "Month", "id": "Month"}, ...]
    """
    return [{"name": name, "id": name} for name in column_names]


# ============================================================================
# HELPER 7 — CONDITIONAL STYLE RULES FOR TABLE ROWS AND CELLS
# ============================================================================

def yoy_colour_style(yoy_column_names):
    """
    Returns Dash DataTable conditional style rules that colour YoY% cells:
      Green  when the cell contains "+"  → positive growth
      Red    when the cell contains "-"  → decline

    This works because format_value(..., "pct+") always produces "+X.XX%" or "-X.XX%",
    so we can detect the sign by checking for the "+" or "-" character in the string.
    """
    style_rules = []

    for col_name in yoy_column_names:
        # Rule: green for any cell in this column whose text contains "+"
        style_rules.append({
            "if": {
                "filter_query": f'{{{col_name}}} contains "+"',
                "column_id": col_name,
            },
            "color": COLOUR_POSITIVE,
            "fontWeight": "bold",
        })
        # Rule: red for any cell in this column whose text contains "-"
        style_rules.append({
            "if": {
                "filter_query": f'{{{col_name}}} contains "-"',
                "column_id": col_name,
            },
            "color": COLOUR_NEGATIVE,
            "fontWeight": "bold",
        })

    return style_rules


def total_row_style(total_number_of_rows):
    """
    Makes the last row bold with a shaded background — the Total row.
    Dash DataTable uses 0-based row indices, so the last row is at index (n - 1).
    """
    return [{
        "if": {"row_index": total_number_of_rows - 1},
        "fontWeight": "bold",
        "backgroundColor": "#f0f2f6",
    }]


def actual_row_style(actual_row_indices):
    """
    Applies a grey background to rows that contain real actuals (not projections).
    Used in Visual 03 to let the user see at a glance which months are actual vs projected.

    actual_row_indices: list of integer row positions (0-based) that are actuals.
    """
    style_rules = []
    for row_index in actual_row_indices:
        style_rules.append({
            "if": {"row_index": row_index},
            "backgroundColor": "#e8e8e8",
        })
    return style_rules


# ============================================================================
# VISUAL 01 — PREVIOUS YEAR MONTHLY SUMMARY (FY25: Jun 2024 – May 2025)
# ============================================================================

def build_visual_01(py_monthly):
    """
    Builds a table showing the full prior fiscal year month by month.

    Columns: Month | Net Revenue | Volume | ADV | Yield | Rev % | Vol %
      Rev %  = this month's revenue as a % of the full-year total
      Vol %  = this month's volume  as a % of the full-year total
    A bold Total row is appended at the bottom.
    """
    # Calculate the full-year totals that the % columns reference
    total_revenue = py_monthly["Revenue"].sum()
    total_volume  = py_monthly["Volume"].sum()
    total_opdays  = py_monthly["OpDays"].sum()

    # Add share-of-total percentage columns to the monthly table
    py_with_pct = py_monthly.copy()
    py_with_pct["Rev_Pct"] = py_with_pct["Revenue"] / total_revenue * 100
    py_with_pct["Vol_Pct"] = py_with_pct["Volume"]  / total_volume  * 100

    # Compute totals for the Total row
    total_yield = total_revenue / total_volume if total_volume else np.nan
    total_adv   = total_volume  / total_opdays if total_opdays else np.nan

    total_row_df = pd.DataFrame([{
        "YM":      "Total",
        "Revenue": total_revenue,
        "Volume":  total_volume,
        "OpDays":  total_opdays,
        "Yield":   total_yield,
        "ADV":     total_adv,
        "Rev_Pct": 100.0,
        "Vol_Pct": 100.0,
    }])

    # Combine monthly rows + Total row into one display DataFrame
    py_display = pd.concat([py_with_pct, total_row_df], ignore_index=True)

    # Convert each row into a dict with display-formatted strings
    table_rows = []
    for _, row in py_display.iterrows():
        table_rows.append({
            "Month":       row["YM"],
            "Net Revenue": format_value(row["Revenue"], "rev"),
            "Volume":      format_value(row["Volume"],  "vol"),
            "ADV":         format_value(row["ADV"],     "adv"),
            "Yield":       format_value(row["Yield"],   "yld"),
            "Rev %":       format_value(row["Rev_Pct"], "pct"),
            "Vol %":       format_value(row["Vol_Pct"], "pct"),
        })

    num_rows  = len(table_rows)
    col_names = ["Month", "Net Revenue", "Volume", "ADV", "Yield", "Rev %", "Vol %"]

    table = make_table(
        rows=table_rows,
        column_definitions=column_defs(col_names),
        conditional_styles=total_row_style(num_rows),
    )

    return html.Div([
        html.H6("Visual 01 — Previous Year  (FY25: Jun 2024 – May 2025)",
                className="fw-bold mt-4 mb-2"),
        table,
        html.Hr(),
    ])


# ============================================================================
# VISUAL 02 — CURRENT YEAR YTD WITH NORMALISATION AND YoY% (FY26)
# ============================================================================

def build_visual_02(cy_monthly):
    """
    Builds a table showing FY26 month-by-month actuals with YoY% comparisons.

    Columns: Month | Net Revenue | Volume | ADV | Yield |
             Norm Revenue | Norm Volume | Norm Yield | Rev YoY% | Vol YoY% | Yield YoY%

    YoY% cells are colour-coded: green = growing, red = declining.
    Normalised columns adjust for operating-day differences between FY25 and FY26.
    """
    # Sum across all CY months for the Total row
    total_cy_rev   = cy_monthly["Revenue"].sum()
    total_cy_vol   = cy_monthly["Volume"].sum()
    total_cy_opd   = cy_monthly["OpDays"].sum()
    total_norm_rev = cy_monthly["NormRevenue"].sum()
    total_norm_vol = cy_monthly["NormVolume"].sum()
    total_py_rev   = cy_monthly["PY_Rev"].sum()
    total_py_vol   = cy_monthly["PY_Vol"].sum()

    total_cy_yld   = total_cy_rev   / total_cy_vol   if total_cy_vol   else np.nan
    total_norm_yld = total_norm_rev / total_norm_vol  if total_norm_vol else np.nan
    total_py_yld   = total_py_rev   / total_py_vol    if total_py_vol   else np.nan

    total_rev_yoy   = (total_cy_rev  / total_py_rev - 1) * 100 if total_py_rev else np.nan
    total_vol_yoy   = (total_cy_vol  / total_py_vol - 1) * 100 if total_py_vol else np.nan
    total_yield_yoy = (total_cy_yld  / total_py_yld - 1) * 100 if total_py_yld else np.nan

    total_row_df = pd.DataFrame([{
        "YM":          "Total",
        "Revenue":     total_cy_rev,
        "Volume":      total_cy_vol,
        "OpDays":      total_cy_opd,
        "ADV":         total_cy_vol / total_cy_opd if total_cy_opd else np.nan,
        "Yield":       total_cy_yld,
        "NormRevenue": total_norm_rev,
        "NormVolume":  total_norm_vol,
        "NormYield":   total_norm_yld,
        "RevYoY":      total_rev_yoy,
        "VolYoY":      total_vol_yoy,
        "YieldYoY":    total_yield_yoy,
    }])

    cy_display = pd.concat([cy_monthly, total_row_df], ignore_index=True)

    table_rows = []
    for _, row in cy_display.iterrows():
        table_rows.append({
            "Month":        row["YM"],
            "Net Revenue":  format_value(row["Revenue"],     "rev"),
            "Volume":       format_value(row["Volume"],      "vol"),
            "ADV":          format_value(row["ADV"],         "adv"),
            "Yield":        format_value(row["Yield"],       "yld"),
            "Norm Revenue": format_value(row["NormRevenue"], "rev"),
            "Norm Volume":  format_value(row["NormVolume"],  "vol"),
            "Norm Yield":   format_value(row["NormYield"],   "yld"),
            "Rev YoY%":     format_value(row["RevYoY"],      "pct+"),
            "Vol YoY%":     format_value(row["VolYoY"],      "pct+"),
            "Yield YoY%":   format_value(row["YieldYoY"],    "pct+"),
        })

    num_rows  = len(table_rows)
    yoy_cols  = ["Rev YoY%", "Vol YoY%", "Yield YoY%"]
    col_names = [
        "Month", "Net Revenue", "Volume", "ADV", "Yield",
        "Norm Revenue", "Norm Volume", "Norm Yield",
        "Rev YoY%", "Vol YoY%", "Yield YoY%",
    ]

    table = make_table(
        rows=table_rows,
        column_definitions=column_defs(col_names),
        conditional_styles=yoy_colour_style(yoy_cols) + total_row_style(num_rows),
    )

    return html.Div([
        html.H6("Visual 02 — Current Year  (FY26: Jun 2025 – May 2026 YTD)",
                className="fw-bold mt-4 mb-2"),
        table,
        html.P(
            "Norm = actual × (PY opdays / CY opdays)  |  YoY% = (CY – PY same calendar month) / PY",
            className="text-muted small mt-1",
        ),
        html.Hr(),
    ])


# ============================================================================
# VISUAL 03 — FULL FY26: ACTUALS + PROJECTIONS
# ============================================================================

def build_visual_03(cy_monthly, projections_df, cy_actual_set):
    """
    Shows all 12 FY26 months in a single table:
      - Months with real data  → grey background (actuals)
      - Future months          → white background (projections)
      - A Total row at the bottom sums revenue + volume across all months

    Parameters
    ----------
    cy_monthly     : monthly summary DataFrame for FY26 actuals
    projections_df : DataFrame with Proj_Rev, Proj_Vol, Proj_Yld for future months
    cy_actual_set  : set of "YYYY-MM" strings that have real CY data (Revenue > 0)
    """
    # Build fast lookup tables indexed by "YYYY-MM"
    actual_lookup     = cy_monthly.set_index("YM")
    projection_lookup = projections_df.set_index("YM") if not projections_df.empty else pd.DataFrame()

    table_rows         = []    # one dict per month that has data
    actual_row_indices = []    # positions of actual rows (for grey shading)

    # Running totals — we add as we go so we don't need to parse the formatted strings later
    total_revenue = 0.0
    total_volume  = 0.0

    for month in FY26_MONTHS:
        if month in cy_actual_set:
            # This month has real actuals
            row_data  = actual_lookup.loc[month]
            month_rev = row_data["Revenue"]
            month_vol = row_data["Volume"]
            month_yld = row_data["Yield"]
            actual_row_indices.append(len(table_rows))   # record position for grey shading

        elif not projection_lookup.empty and month in projection_lookup.index:
            # This month has a projection only
            row_data  = projection_lookup.loc[month]
            month_rev = row_data["Proj_Rev"]
            month_vol = row_data["Proj_Vol"]
            month_yld = row_data["Proj_Yld"]

        else:
            continue   # no data and no projection — skip

        total_revenue += month_rev
        total_volume  += month_vol

        table_rows.append({
            "Month":       month,
            "Net Revenue": format_value(month_rev, "rev"),
            "Volume":      format_value(month_vol, "vol"),
            "Yield":       format_value(month_yld, "yld"),
        })

    if not table_rows:
        return dbc.Alert("No FY26 data available for Visual 03.", color="info")

    # Append the Total row
    total_yield = total_revenue / total_volume if total_volume else np.nan
    table_rows.append({
        "Month":       "Total",
        "Net Revenue": format_value(total_revenue, "rev"),
        "Volume":      format_value(total_volume,  "vol"),
        "Yield":       format_value(total_yield,   "yld"),
    })

    num_rows       = len(table_rows)
    num_actuals    = len(actual_row_indices)
    col_names      = ["Month", "Net Revenue", "Volume", "Yield"]

    caption = (
        f"Grey = actuals ({num_actuals} month{'s' if num_actuals != 1 else ''})  |  "
        "White = projections  |  Formula: (CY norm total / PY attainment) × PY month value"
    )

    table = make_table(
        rows=table_rows,
        column_definitions=column_defs(col_names),
        conditional_styles=actual_row_style(actual_row_indices) + total_row_style(num_rows),
    )

    return html.Div([
        html.H6(
            "Visual 03 — Current Year: Actuals + Projections  (FY26: Jun 2025 – May 2026)",
            className="fw-bold mt-4 mb-2",
        ),
        table,
        html.P(caption, className="text-muted small mt-1"),
        html.Hr(),
    ])


# ============================================================================
# LAYOUT — SIDEBAR + MAIN CONTENT AREA
# ============================================================================

sidebar = dbc.Card([
    html.Div([
        html.Span("GSM Tool", className="fw-bold fs-6 text-primary"),
        html.Br(),
        html.Span("Performance Dashboard v2", className="text-muted small"),
    ], className="mb-3"),
    html.Hr(className="my-2"),

    # Fuel surcharge toggle
    html.Label("Fuel Surcharge", className="fw-semibold small mb-1"),
    dcc.RadioItems(
        id="fuel-radio",
        options=[
            {"label": " Include", "value": "Include"},
            {"label": " Exclude", "value": "Exclude"},
        ],
        value="Include",
        className="mb-3",
    ),

    html.Hr(className="my-2"),

    # OPCO filter (e.g. FXG, FXED)
    html.Label(f"OPCO  ({len(all_opcos)} total)", className="fw-semibold small mb-1"),
    dcc.Dropdown(
        id="opco-dropdown",
        options=all_opcos,
        multi=True,
        placeholder=f"All ({len(all_opcos)})",
        className="mb-2",
    ),

    # SVP filter
    html.Label("SVP", className="fw-semibold small mb-1"),
    dcc.Dropdown(
        id="svp-dropdown",
        options=all_svps,
        multi=True,
        placeholder=f"All ({len(all_svps)})",
        className="mb-2",
    ),

    # VP filter — options change based on which SVPs are selected
    html.Label("VP", className="fw-semibold small mb-1"),
    dcc.Dropdown(
        id="vp-dropdown",
        multi=True,
        placeholder="All",
        className="mb-2",
    ),

    # Director / Manager filter — options change based on SVP + VP selection
    html.Label("DIR / MGR", className="fw-semibold small mb-1"),
    dcc.Dropdown(
        id="dir-dropdown",
        multi=True,
        placeholder="All",
    ),
], body=True, style={
    "position": "sticky",
    "top": "10px",
    "maxHeight": "calc(100vh - 30px)",
    "overflowY": "auto",
    "fontSize": "13px",
})

app.layout = dbc.Container([
    dbc.Row(
        dbc.Col(html.H4("GSM Tool — Performance Dashboard v2", className="text-primary my-3"))
    ),
    dbc.Row([
        dbc.Col(sidebar, width=3),
        dbc.Col(
            html.Div(id="dashboard-content", style={"paddingTop": "8px"}),
            width=9,
        ),
    ]),
], fluid=True)


# ============================================================================
# CALLBACK 1 — UPDATE VP DROPDOWN WHEN SVP CHANGES
# ============================================================================

@app.callback(
    Output("vp-dropdown", "options"),
    Output("vp-dropdown", "value"),
    Input("svp-dropdown", "value"),
)
def update_vp_options(selected_svps):
    """
    When the user picks one or more SVPs, narrow the VP list to only
    show VPs who report to those SVPs.
    If no SVPs are selected, show all VPs.
    """
    svp_filter = selected_svps if selected_svps else all_svps

    vp_options = sorted(
        all_data[all_data["SLS_SVP_EMP_NM"].isin(svp_filter)]["SLS_VP_EMP_NM"].dropna().unique()
    )

    return vp_options, None   # None resets the VP selection


# ============================================================================
# CALLBACK 2 — UPDATE DIRECTOR DROPDOWN WHEN SVP OR VP CHANGES
# ============================================================================

@app.callback(
    Output("dir-dropdown", "options"),
    Output("dir-dropdown", "value"),
    Input("svp-dropdown", "value"),
    Input("vp-dropdown",  "value"),
)
def update_dir_options(selected_svps, selected_vps):
    """
    Narrows the Director list based on the currently selected SVPs and VPs.
    Falls back to "all" for any level not explicitly selected.
    """
    svp_filter = selected_svps if selected_svps else all_svps

    available_vps = sorted(
        all_data[all_data["SLS_SVP_EMP_NM"].isin(svp_filter)]["SLS_VP_EMP_NM"].dropna().unique()
    )
    vp_filter = selected_vps if selected_vps else available_vps

    dir_options = sorted(
        all_data[
            all_data["SLS_SVP_EMP_NM"].isin(svp_filter) &
            all_data["SLS_VP_EMP_NM"].isin(vp_filter)
        ]["SLS_DIR_MGR_NM"].dropna().unique()
    )

    return dir_options, None   # None resets the Director selection


# ============================================================================
# CALLBACK 3 — REBUILD ALL THREE VISUALS WHEN ANY FILTER CHANGES
# ============================================================================

@app.callback(
    Output("dashboard-content", "children"),
    Input("fuel-radio",    "value"),
    Input("opco-dropdown", "value"),
    Input("svp-dropdown",  "value"),
    Input("vp-dropdown",   "value"),
    Input("dir-dropdown",  "value"),
)
def update_dashboard(fuel_choice, selected_opcos, selected_svps, selected_vps, selected_dirs):
    """
    Main callback — runs every time a filter changes.

    Steps:
      1. Pick the revenue column based on the fuel toggle
      2. Resolve cascaded filter selections (if nothing selected → treat as "all")
      3. Filter all_data down to the selected slice
      4. Split into FY25 (prior year) and FY26 (current year)
      5. Aggregate into monthly summaries
      6. Add prior-year columns + normalisation to the CY table
      7. Calculate projections for future FY26 months
      8. Build and return all three visuals
    """

    # ── Step 1: Revenue column ────────────────────────────────────────────────
    if fuel_choice == "Include":
        revenue_column = "NET_REV_AMT"       # revenue including fuel surcharge
    else:
        revenue_column = "NET_REV_AMT_WOF"   # revenue excluding fuel surcharge

    # ── Step 2: Resolve cascaded filters ─────────────────────────────────────
    # If the user selected nothing at a level, fall back to all options at that level.
    svp_filter = selected_svps if selected_svps else all_svps

    available_vps = sorted(
        all_data[all_data["SLS_SVP_EMP_NM"].isin(svp_filter)]["SLS_VP_EMP_NM"].dropna().unique()
    )
    vp_filter = selected_vps if selected_vps else available_vps

    available_dirs = sorted(
        all_data[
            all_data["SLS_SVP_EMP_NM"].isin(svp_filter) &
            all_data["SLS_VP_EMP_NM"].isin(vp_filter)
        ]["SLS_DIR_MGR_NM"].dropna().unique()
    )
    dir_filter  = selected_dirs  if selected_dirs  else available_dirs
    opco_filter = selected_opcos if selected_opcos else all_opcos

    # ── Step 3: Apply filters ─────────────────────────────────────────────────
    filtered_data = all_data[
        all_data["OPCO_BREAKOUT"].isin(opco_filter) &
        all_data["SLS_SVP_EMP_NM"].isin(svp_filter) &
        all_data["SLS_VP_EMP_NM"].isin(vp_filter) &
        all_data["SLS_DIR_MGR_NM"].isin(dir_filter)
    ].copy()

    if filtered_data.empty:
        return dbc.Alert("No data matches the selected filters.", color="warning")

    # ── Step 4: Split into FY25 and FY26 ─────────────────────────────────────
    # FY25 = Jun 2024 through May 2025
    prior_year_data = filtered_data[
        ((filtered_data["Year"] == 2024) & (filtered_data["MonthNum"] >= 6)) |
        ((filtered_data["Year"] == 2025) & (filtered_data["MonthNum"] <  6))
    ]

    # FY26 = Jun 2025 through May 2026
    current_year_data = filtered_data[
        ((filtered_data["Year"] == 2025) & (filtered_data["MonthNum"] >= 6)) |
        ((filtered_data["Year"] == 2026) & (filtered_data["MonthNum"] <  6))
    ]

    if prior_year_data.empty or current_year_data.empty:
        return dbc.Alert(
            "Not enough data for the FY25 / FY26 period split. "
            "Ensure your CSV files span both fiscal years.",
            color="warning",
        )

    # ── Step 5: Monthly summaries ─────────────────────────────────────────────
    py_monthly = aggregate_by_month(prior_year_data,   revenue_column)
    cy_monthly = aggregate_by_month(current_year_data, revenue_column)

    # ── Step 6: Attach PY columns + normalisation to the CY table ────────────
    cy_monthly = add_prior_year_columns(cy_monthly, py_monthly)

    # ── Step 7: Projections ───────────────────────────────────────────────────
    projections_df, last_actual_month = calculate_projections(cy_monthly, py_monthly)

    # Set of months that have real CY data (Revenue > 0)
    cy_actual_set = set(cy_monthly[cy_monthly["Revenue"] > 0]["YM"])

    # ── Step 8: Build and return all three visuals ────────────────────────────
    visual_01 = build_visual_01(py_monthly)
    visual_02 = build_visual_02(cy_monthly)
    visual_03 = build_visual_03(cy_monthly, projections_df, cy_actual_set)

    return html.Div([visual_01, visual_02, visual_03])


# ============================================================================
# RUN THE APP
# ============================================================================

if __name__ == "__main__":
    app.run(debug=True, port=8051)

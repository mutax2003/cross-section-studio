# Workbook format

## Required sheets (native profile)

### Collars

| Column | Type | Notes |
|--------|------|--------|
| `hole_id` | text | Unique borehole ID |
| `easting` | float | Map X |
| `northing` | float | Map Y |
| `elevation` | float | Collar RL |
| `total_depth` | float | Metres below collar |

Optional: `elevation_datum`, `inclination_deg`, `azimuth_deg`.

### Lithology

| Column | Type | Notes |
|--------|------|--------|
| `hole_id` | text | Must match a collar |
| `from_depth` | float | Metres below collar |
| `to_depth` | float | Metres below collar |
| `lithology_code` | text | Prefer USGS-style codes (Sand, Clay, Topsoil, …) |

Optional: `hatch_pattern`, `unit_order` (1 = shallowest; required when the same code repeats in one hole).

## Optional sheets

| Sheet | Columns | Purpose |
|-------|---------|---------|
| **Water** | `hole_id`, `depth` **or** `elevation_masl` (not both) | Groundwater markers. Use `depth` (below collar) or `elevation_masl` (RL). Optional: `series_id`, `series_label`, `color`, `marker` for multi-date snapshots |
| **Screens** | `hole_id`, `from_depth`, `to_depth` | Screen interval hatch bands (consulting layout) |
| **Gradients** | `hole_id`, `direction` (`up` / `down`) | Vertical gradient arrows (consulting layout) |
| **Deviations** | `hole_id`, `depth`, `inclination_deg`, `azimuth_deg` | Deviated stick paths |
| **Correlations** | `left_hole_id`, `right_hole_id`, `left_unit_order`, `right_unit_order` | Manual unit pairing |
| **Faults** | `name`, `x_profile`, `elevation` | Profile-plane fault traces |
| **Unconformities** | `name`, `x_profile`, `elevation` | Profile-plane surfaces |
| **Environmental** | `hole_id`, `parameter`, `value`, `depth` **or** `from_depth`+`to_depth` | Lab/screening samples (e.g. chloride at 3.5 m). Optional: `unit` (e.g. `mg/L`) |

Sheet names are matched case-insensitively.

## Field export profile

Single `Lithology` sheet with `Label`, `Depth` (e.g. `0.00-2.00m`), `Lithology`, `Lat`, `Long`. Converted to UTM on import; elevation may use a profile placeholder until surveyed RL is provided.

## Lithology styles

Canonical colors come from `data/BH Log Lithology Legend.xlsx` (loaded at runtime by `constants.py`). Hatches live in `USGS_LITHOLOGY_HATCHES`. A JSON cache (`data/bh_log_lithology_legend.json`) is used when the Excel file is absent (e.g. frozen builds); regenerate with `python scripts/convert_bh_log_legend.py`. Overrides can be saved from the app fill-style editor (`data/lithology_styles.json`).

Chloride average concentrations for Advantage Phase 2 transects A–A′ and B–B′ are provided in `data/Cross_Section_Chlorides.xlsx` and loaded via `advantage_p2_reference.chlorides.load_chloride_readings()`.

## Multi-tab input template

For field teams, download the multi-tab template from the app welcome card, or regenerate with `python scripts/build_input_template.py` (writes `data/Cross_Section_Input_Template.xlsx`, or `_v3.xlsx` if the primary file is locked in Excel):

| Tab | Purpose |
|-----|---------|
| **Instructions** | Quick start, tab guide, lithology codes, tips |
| **Project** | Client / figure metadata (seeds consulting title block on upload) |
| **Collars** | Required — one row per borehole |
| **Lithology** | Required — depth intervals (`from_depth` / `to_depth` below collar) |
| **Water** | Optional — groundwater `depth` **or** `elevation_masl` (not both) |
| **Environmental** | Optional — lab/field parameters at point depth or interval |
| **Screens** | Optional — screened intervals (consulting hatch) |
| **Gradients** | Optional — vertical gradient `up` / `down` |
| **Example** | Filled MW-01…03 demo (reference only — not parsed) |
| **Data Entry** | Compatibility sheet (PROJECT metadata for auto-detect) |

Upload the workbook directly. Named data tabs are preferred; when `Collars` / `Lithology` are absent, the parser can still read geology sections from **Data Entry**.

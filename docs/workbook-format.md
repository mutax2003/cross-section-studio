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
| **Water** | `hole_id`, `depth` | Groundwater markers (depth below collar). Optional: `series_id`, `series_label`, `color`, `marker` for multi-date snapshots |
| **Screens** | `hole_id`, `from_depth`, `to_depth` | Screen interval hatch bands (consulting layout) |
| **Gradients** | `hole_id`, `direction` (`up` / `down`) | Vertical gradient arrows (consulting layout) |
| **Deviations** | `hole_id`, `depth`, `inclination_deg`, `azimuth_deg` | Deviated stick paths |
| **Correlations** | `left_hole_id`, `right_hole_id`, `left_unit_order`, `right_unit_order` | Manual unit pairing |
| **Faults** | `name`, `x_profile`, `elevation` | Profile-plane fault traces |
| **Unconformities** | `name`, `x_profile`, `elevation` | Profile-plane surfaces |
| **Environmental** | `hole_id`, `from_depth`, `to_depth`, `parameter`, `value` | Screening hits |

Sheet names are matched case-insensitively.

## Field export profile

Single `Lithology` sheet with `Label`, `Depth` (e.g. `0.00-2.00m`), `Lithology`, `Lat`, `Long`. Converted to UTM on import; elevation may use a profile placeholder until surveyed RL is provided.

## Lithology styles

Canonical colors and hatches live in `constants.py` (`USGS_LITHOLOGY_COLORS`, `USGS_LITHOLOGY_HATCHES`). Overrides can be saved from the app fill-style editor (`data/lithology_styles.json`).

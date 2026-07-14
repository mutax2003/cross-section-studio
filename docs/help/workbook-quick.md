# Workbook quick reference

Upload a multi-tab Excel workbook. Sheet names are matched case-insensitively.

## Required sheets

### Collars

One row per borehole: `hole_id`, `easting`, `northing`, `elevation`, `total_depth`.

Optional: `elevation_datum`, `inclination_deg`, `azimuth_deg`.

### Lithology

Depth intervals below collar: `hole_id`, `from_depth`, `to_depth`, `lithology_code`.

Optional: `hatch_pattern`, `unit_order` (needed when the same code repeats in one hole).

## Optional sheets

| Sheet | Purpose |
|-------|---------|
| **Water** | Groundwater markers (`depth` **or** `elevation_masl`, not both) |
| **Screens** | Screen interval hatch bands |
| **Gradients** | Vertical gradient arrows (`up` / `down`) |
| **Environmental** | Lab/screening values at depth or interval |
| **Deviations** | Deviated stick survey points |
| **Correlations** | Manual unit pairing between holes |
| **Faults** / **Unconformities** | Profile-plane overlays |

The multi-tab input template also includes **Project** (title-block metadata), **Instructions**, and an **Example** tab (reference only — not parsed).

Full column lists, field-export profiles, and advanced sheets: [workbook format](../workbook-format.md).

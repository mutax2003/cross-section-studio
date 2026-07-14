# Workbook & data entry

**Enter data in Excel** (multi-tab workbook), then **upload** it via **Upload Excel workbook** in the sidebar. Sheet names are matched case-insensitively.

## Enter in Excel, then upload

1. Download **Download template (data entry)** from the welcome card or sidebar Data source section.
2. Fill **Collars** and **Lithology** in Excel (optional sheets below).
3. Upload the saved `.xlsx` with **Upload Excel workbook** — same ingest path as any other workbook.
4. Use File → **Load sample project** only when you want the built-in demo (skips template fill).

A legacy **Data Entry** sheet still imports for compatibility; prefer the named tabs in the template.

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

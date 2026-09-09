# Workbook & data entry

**Enter data in Excel** (multi-tab workbook), then **upload** it via **Upload Excel workbook** in the sidebar. Sheet names are matched case-insensitively.

## Enter in Excel, then upload

1. Download **Download template (data entry)** from the welcome card or sidebar Data source section.
2. Fill **Collars** and **Lithology** in Excel (optional sheets below).
3. Upload the saved `.xlsx` with **Upload Excel workbook** — same ingest path as any other workbook.
4. Use File → **Load sample project** (or `Ctrl+Shift+O`) only when you want the built-in demo.

A legacy **Data Entry** sheet still imports for compatibility; prefer the named tabs in the template.

## Required sheets

### Collars

One row per borehole: `hole_id`, `easting`, `northing`, `elevation`, `total_depth`.

Optional: `elevation_datum`, `inclination_deg`, `azimuth_deg`, `stick_up_m`.

### Lithology

Depth intervals below collar: `hole_id`, `from_depth`, `to_depth`, `lithology_code`.

Optional: `hatch_pattern`, `unit_order` (needed when the same code repeats in one hole).

## Optional sheets

| Sheet | Purpose |
|-------|---------|
| **Water** | Groundwater markers (`depth` **or** `elevation_masl`, not both). Optional series columns for multi-date snapshots. |
| **Screens** | Screen interval hatch bands (consulting layout) |
| **Gradients** | Vertical gradient arrows (`up` / `down`) |
| **Environmental** | Lab/screening values at depth or interval — pick parameters on **Configure** |
| **Sections** | `section_label` + ordered `hole_ids` — seeds Configure **Multi-transect batch ZIP** (also **Load from workbook Sections**) |
| **Field Data** | Field-export sheet with `OVA` / `EC` → environmental readings (`OVA` / `EC`); **not** used for stratigraphy |
| **Deviations** | Deviated stick survey points |
| **Correlations** | Manual unit pairing between holes |
| **Faults** / **Unconformities** | Profile-plane overlays |

The multi-tab input template also includes **Project** (title-block metadata), **Instructions**, and an **Example** tab (reference only — not parsed).

Full column lists, field-export profiles, and advanced sheets: [workbook format](../workbook-format.md).

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

Optional: `elevation_datum`, `inclination_deg`, `azimuth_deg`, `stick_up_m` (metres above collar for MW stick-up). `inclination_deg` is degrees from vertical (0 = straight down); a negative value is read as a mining-style dip (−90 = straight down, −60 = 30° off vertical). A collar angle with no Deviations rows draws a straight inclined hole; a Deviations row at depth 0 overrides the collar angles.

### Lithology

| Column | Type | Notes |
|--------|------|--------|
| `hole_id` | text | Must match a collar |
| `from_depth` | float | Metres below collar |
| `to_depth` | float | Metres below collar |
| `lithology_code` | text | Prefer USGS-style codes (Sand, Clay, Topsoil, …) |

Lithology colours and hatches follow the scheme agreed on 1 Oct 2026: one base colour per soil group (clay `#967259`, loam `#C68642`, sand `#FFE39F`, rock `#4C516D`) with a hatch for the secondary component only (sandy = dots — including Sand itself — silty = 45° lines, gravel = cobbles, `+` = the clay-loam mixes Clay Loam and Silty Clay Loam), at the sparse densities of Sheryl's CAD template (Cross_Section_Litho_Legend_261002); Clay, Silt, Loam, Topsoil, Organics and Coal are plain (Coal is the only black unit). Full table: `data/bh_log_lithology_legend.json` + `constants.py`. Survey123 swatches: `python scripts/export_lithology_swatches.py --width W --height H --zip out.zip`.

Optional: `hatch_pattern`, `unit_order` (1 = shallowest).

**Automatic `unit_order`.** Layers correlate between holes on `unit_order` + `lithology_code` (or the code alone when it is unique in the hole). When a hole repeats a code and some rows have no `unit_order`, the app (sidebar **Number layers down each hole automatically**, on by default) lines every hole up into one **site-wide** unit sequence and gives each blank row its unit's number, so a unit missing from one hole does not shift the deeper numbers. The result does not depend on sheet row order. Explicit values are kept; an automatic number that clashes with one in the same hole is renumbered and reported. Validate lists the sequence used and any hole whose layers could not be lined up (those pinch out — set `unit_order` to correlate them).

**Lithology QA warnings** (shown in Validate; like other warnings, the first two need **Generate even if data checks found warnings** ticked — logging gaps never do):

| Warning | Meaning |
|---------|---------|
| unit_order runs against depth | A layer has a larger `unit_order` than the one below it. The fence judges crossings on depth, so those numbered matches become pinch-outs — check the numbering. |
| Zero-thickness interval | `from_depth` = `to_depth`; the row is ignored in the fence. Fix `to_depth` or delete it. |
| Nothing logged / gap | No row covers a depth range — top of hole, between rows, or down to total depth (no recovery / not logged). The column draws it grey with a **Not logged** legend key, and the fence closes an internal gap between the units above and below. Generate is not blocked. A logged `No Recovery` code is a lithology of its own (white). |

## Optional sheets

| Sheet | Columns | Purpose |
|-------|---------|---------|
| **Water** | `hole_id`, `depth` **or** `elevation_masl` (not both) | Groundwater markers. Use `depth` (along the hole, below ground by default) or `elevation_masl` (RL). Optional `depth_reference`: `bgs` (default; also `mbgs`, `bgl`, `ground`) or `btoc` (also `toc`, `mbtoc`, `top of casing`) — btoc heads are collar + Collars `stick_up_m` − depth (blank stick-up = 0 m, with a Validate warning). Surveyed holes (Deviations sheet) convert water, screen and sample depths to true vertical depth. Optional: `status` (`measured` default, `dry`, or `nm`), `series_id`, `series_label`, `connect_group` (nest id — only rows with the same group connect), `color`, `marker`. Optional **LNAPL** columns (at the end): `product_thickness_m` (apparent free-product thickness, m ≥ 0), `product_density` (specific gravity 0.5–<1.0, default 0.75; ≥ 1.0 is rejected as DNAPL) and `depth_to_product` (alternative to the thickness: depth to the product top on the same datum as `depth`; thickness = depth − depth_to_product, error if deeper than `depth`). With product, `depth` / `elevation_masl` is the measured **water** level (oil/water interface) and the plotted head is corrected: water elevation + thickness × SG (vertical thickness on surveyed holes). Rows with `dry`/`nm` may omit depth/masl. Up to **4** series can be plotted; choose them on **Configure**. Optional `date` (monitoring date: an Excel date cell, an Excel serial, or ISO text `2026-04-02` / `2026/04/02`; blank = undated; day-first text such as `02/04/2026` is rejected as ambiguous) — keep each series to one monitoring event: Validate warns when a series / `connect_group` nest spans more than 7 days (*heads from different dates are not one potentiometric surface*) and notes a dated + undated mix; when one hole has several readings in a series / nest the **latest date** is plotted and used for gradients, nested iv and the plan-view flow fit (undated duplicates: last row, as before); an unlabelled series from one event shows its date in the legend; flow-fit and vertical-gradient notes name the reading dates. Schematic water lines are **not** a potentiometric surface. |
| **Screens** | `hole_id`, `from_depth`, `to_depth` | Screen interval hatch bands (consulting layout) |
| **Gradients** | `hole_id`, `direction` (`up` / `down`) | Vertical gradient arrows (consulting layout) |
| **Deviations** | `hole_id`, `depth`, `inclination_deg`, `azimuth_deg` | Deviated stick paths |
| **Correlations** | `left_hole_id`, `right_hole_id`, `left_unit_order`, `right_unit_order` | Manual unit pairing |
| **Faults** | `name`, `x_profile`, `elevation` | Profile-plane fault traces |
| **Unconformities** | `name`, `x_profile`, `elevation` | Profile-plane surfaces |
| **Environmental** | `hole_id`, `parameter`, `value`, `depth` **or** `from_depth`+`to_depth` | Lab/screening samples (e.g. chloride at 3.5 m). Optional: `unit` (e.g. `mg/L`), `value_label` (figure text), `label_color` — drop-down `green` / `red` / `black` / `orange`, one fixed colour each (blue is reserved for groundwater). Non-detects: type `<5` in `value` (also `< 5`, `<RDL 5`, `ND (5)`, `n.d. 5`; a bare `ND` needs its limit), or the limit in `value` and `<` / `ND` in the optional `qualifier` column (a `detected` yes/no column is accepted too) — the figure prints `<5`; Validate warns when a non-detect's limit is above the parameter's `green_max` and when one parameter is reported in more than one unit (values are never converted). Optional `green_max` / `orange_max` (also `yellow_max`): that parameter's own threshold limits in its own units — fill them on one row per parameter (e.g. Chloride 100 / 250, Benzene 0.5 / 5) and Configure's threshold editor starts from them. Select parameters on **Configure**; rows without a `label_color` use the Configure setting (black, or green/orange/red thresholds, resolved per parameter). Optional `date` (sampling date, same formats as Water): when one hole / parameter / depth has samples on several dates the latest is plotted, and Validate and the section notes say so (*BH-2 Benzene at 3 m: 3 sampling dates, latest (2026-05-01) shown*); undated duplicates plot as before. |
| **Sections** | `section_label`, `hole_ids` | Named section lines. Configure shows a **Section (from workbook Sections tab)** drop-down that previews any row; the rows also seed **Multi-transect batch ZIP** for exporting every section. Rows that name unknown holes are reported on upload. `hole_ids` are ordered holes separated by comma, semicolon, or `→` (e.g. `MW-01, MW-02, MW-03` or `MW-01→MW-02→MW-03`). Each valid row (≥2 known collars) seeds a line `Label \| h1, h2, …`. |

Sheet names are matched case-insensitively.

## Field export profile

Single `Lithology` sheet with `Label`, `Depth` (e.g. `0.00-2.00m`), `Lithology`, `Lat`, `Long`. Converted to UTM on import; elevation may use a profile placeholder until surveyed RL is provided.

Optional **Field Data** sheet (same workbook): `Label` (or `hole_id`), `Depth` interval, plus `OVA` and/or `EC`. Those columns become environmental readings with parameters `OVA` / `EC` (not stratigraphy). Select them on **Configure** like other Environmental parameters.

## Lithology styles

Canonical colors and marks follow Sheryl's CAD template **Cross_Section_Litho_Legend_261002**: colours in `data/bh_log_lithology_legend.json`, hatches in `USGS_LITHOLOGY_HATCHES` (`constants.py`). The old `data/BH Log Lithology Legend.xlsx` hex sheet is a fallback only: it adds codes 261002 does not list, and never recolours a 261002 code. Per-project overrides can be saved from the app fill-style editor (`data/lithology_styles.json`).

Chloride average concentrations for Advantage Phase 2 transects A–A′ and B–B′ are loaded via `advantage_p2_reference.chlorides.load_chloride_readings()`, which reads `data/Cross_Section_Chlorides.xlsx` when present (a client workbook, not tracked in the repo) and otherwise falls back to the packaged fixture `advantage_p2_reference/chloride_readings.json`.

## Multi-tab input template

For field teams, download the multi-tab template from the app welcome card, or regenerate with `python scripts/build_input_template.py` (writes `data/Cross_Section_Input_Template.xlsx`, or `_v3.xlsx` if the primary file is locked in Excel):

| Tab | Purpose |
|-----|---------|
| **Instructions** | Quick start, tab guide, lithology codes, tips |
| **Project** | Client / figure metadata (seeds the consulting title block on upload). Includes `map_scale` (e.g. `1:1000`; consulting sheets print at exactly that scale when it fits the page, else AS SHOWN), `vertical_exaggeration` (a number such as `5` for exactly that VE, or `auto` to fit the page; ignored by the Chemistry columns style, which is locked at 1×) and `figure_preset` |
| **Collars** | Required — one row per borehole |
| **Lithology** | Required — depth intervals (`from_depth` / `to_depth` below collar) |
| **Water** | Optional — groundwater `depth` **or** `elevation_masl` (not both) |
| **Environmental** | Optional — lab/field parameters at point depth or interval |
| **Screens** | Optional — screened intervals (consulting hatch) |
| **Gradients** | Optional — vertical gradient `up` / `down` |
| **Sections** | Optional — `section_label` + `hole_ids` (seeds Configure multi-transect batch) |
| **Example** | Filled MW-01…03 demo (reference only — not parsed) |
| **Data Entry** | Compatibility sheet (PROJECT metadata for auto-detect) |

Upload the workbook directly. Named `Collars` / `Lithology` tabs are preferred; when they are absent (or empty), the parser can still read geology sections from **Data Entry**. For the optional `Water` / `Environmental` / `Screens` / `Gradients` tabs the precedence is reversed: rows in **Data Entry** take priority, and populated native tabs are ignored with a warning.

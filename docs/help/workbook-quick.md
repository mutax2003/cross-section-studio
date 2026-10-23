# Workbook & data entry

**Enter data in Excel** (multi-tab workbook), then **upload** it via **Upload Excel workbook** in the sidebar. Sheet names are matched case-insensitively.

## Enter in Excel, then upload

1. Download **Download template (data entry)** from the welcome card or sidebar **Data** section.
2. Fill **Collars** and **Lithology** in Excel (optional sheets below).
3. Upload the saved `.xlsx` with **Upload Excel workbook** — same ingest path as any other workbook.
4. Use File → **Load sample project** (or `Alt+Shift+O`) only when you want the built-in demo.

A legacy **Data Entry** sheet still imports for compatibility; prefer the named tabs in the template.

## Required sheets

### Collars

One row per borehole: `hole_id`, `easting`, `northing`, `elevation`, `total_depth`.

Optional: `elevation_datum`, `inclination_deg`, `azimuth_deg`, `stick_up_m`.

### Lithology

Depth intervals below collar: `hole_id`, `from_depth`, `to_depth`, `lithology_code`.

Colours and hatches follow the client lithology legend **261023** — use its names (Clay, Silty Clay, Clay Loam, Sandy Clay Loam, Loam, Loamy Sand, Sand, Sand and Gravel, Gravel, Topsoil, Fill, …). Unknown codes draw in grey.

Optional: `hatch_pattern`, `unit_order` (1 = shallowest). You can usually leave `unit_order` blank: layers correlate by code, and when a hole repeats a code the app numbers the layers for you, lining all holes up into one site-wide sequence so the same layer gets the same number in every hole (sidebar **Number layers down each hole automatically**, on by default). Enter it only to force a correlation.

Validate also checks the log:

- **unit_order runs against depth** (a layer numbered higher than the one below it) — those layers are treated as crossing and drawn as pinch-outs; check the numbering.
- **Zero-thickness interval** (`from_depth` = `to_depth`) — ignored in the fence; fix `to_depth` or delete the row.
- **Logging gap** (no row over a depth range) — drawn grey and keyed **Not logged** in the legend.

The first two count as warnings (fix them, or tick **Generate even if data checks found warnings**); logging gaps are listed but never block Generate.

## Optional sheets

| Sheet | Purpose |
|-------|---------|
| **Water** | Groundwater markers (`depth` **or** `elevation_masl`, not both). Optional `status` (`measured` / `dry` / `nm`), series columns for multi-date snapshots, and `depth_reference`: `bgs` (default, depth below ground) or `btoc` (depth below top of casing — head = collar + Collars `stick_up_m` − depth; blank stick-up counts as 0 m and Validate warns). Depths are along the hole: surveyed holes (Deviations) plot water, screens and chemistry at true vertical depth, like the lithology; depth-mode labels show vertical depth below ground. LNAPL wells: `depth` is the depth to water (oil/water interface); add `product_thickness_m` (or `depth_to_product`) and `product_density` (SG, default 0.75) — heads are corrected by thickness × SG. Optional `date` (YYYY-MM-DD or an Excel date): keep one series to one monitoring event — Validate warns when a series spans more than 7 days, and duplicate readings use the latest date. |
| **Screens** | Screen interval hatch bands (consulting layout) |
| **Gradients** | Vertical gradient arrows (`up` / `down`) |
| **Environmental** | Lab/screening values at depth or interval — pick parameters on **Configure** |
| **Sections** | `section_label` + ordered `hole_ids` — seeds Configure **Several section lines (batch ZIP)** (also **Load from workbook Sections**) |
| **Field Data** | Field-export sheet with `OVA` / `EC` → environmental readings (`OVA` / `EC`); **not** used for stratigraphy |
| **Deviations** | Deviated stick survey points |
| **Correlations** | Manual unit pairing between holes |
| **Faults** / **Unconformities** | Profile-plane overlays |

The multi-tab input template also includes **Project** (title-block metadata), **Instructions**, and an **Example** tab (reference only — not parsed). On the Project tab, `map_scale` (e.g. `1:1000`) sets the title-block map scale and `vertical_exaggeration` takes a number (e.g. `5`) or `auto` (fit page).

Full column lists, field-export profiles, and advanced sheets: [workbook format](../workbook-format.md).

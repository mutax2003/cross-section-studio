# Getting started

Cross Section Studio builds borehole **fence diagrams** from an Excel workbook. **Enter geology in Excel**, then **upload** the file in the app — the workbook is the source of truth (geology is not edited in Streamlit widgets).

## Enter in Excel, then upload

1. Download **Download template (data entry)** from the welcome card or sidebar — fill **Collars** and **Lithology** (required).
2. Upload the saved `.xlsx` with **Upload Excel workbook** (filled template, native workbook, or field export with Lat/Long).
3. Continue **Validate → Configure → Generate**.

Fastest path with no prep: **Try sample project** (or `Ctrl+Shift+O`).

## Workflow

1. **Upload** — Template + Excel, or sidebar upload of an existing `.xlsx`.
2. **Validate** — Review parse warnings, lithology codes, groundwater series, and environmental readings. Fix sheet issues before configuring.
3. **Configure** — Choose holes, transect order, layout/style, and overlays (water, screens, gradients, parameters). Resolve correlation preflight if prompted. Optional QA gates:
   - **Block export on polygon overlaps** — stops Generate until overlaps are resolved (consulting presets often enable this).
   - **Warn on correlation gaps** (Advanced) — surfaces unmatched units / pinch-out candidates.
4. **Generate** — Builds the cross-section as **SVG immediately**. Then click **Prepare deliverables** for PNG/PDF and the drafter package (Word, clipboard, report ZIP, project folder). Regenerate after config changes.

Multi-transect packaging is configured under **Multi-transect batch ZIP**, then built from Generate — see [Generate & exports](generate-exports.md).

## Sample project

**Try sample project** (or `Ctrl+Shift+O`) loads the built-in demo workbook so you can walk Validate → Configure → Generate without preparing your own data.

## Input template

**Download template (data entry)** includes required **Collars** / **Lithology**, optional Water, Screens, Gradients, Environmental, **Sections** (named transects for batch ZIP), plus Project metadata and an Example tab (reference only).

Field-export workbooks may include **Field Data** (`OVA` / `EC`) — those become environmental readings, not stratigraphy.

Column details: [Workbook & data entry](workbook-quick.md) and the full [workbook format](../workbook-format.md).

## Before client delivery

Inter-hole fills are a **rule-based 2D fence** (linear contacts, midpoint pinch-outs) — not geostatistical interpolation. Review Configure preflight, overlap warnings, and regenerate after any correlation fix.

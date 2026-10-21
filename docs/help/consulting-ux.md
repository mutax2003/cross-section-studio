# Consulting UX (gINT / Strater / RockWorks workflow)

Cross Section Studio targets **2D fence reporting** — Excel as source of truth, plan view, A–A′ section sheet, then PDF deliverables. It is **not** a 3D modeller (Leapfrog-style).

Inter-hole fills are a **rule-based fence** (linear contacts, midpoint pinch-outs), not geostatistical surfaces. Treat figures as schematic engineering drawings and review correlation / overlap QA before client delivery.

## Workflow map

| Step | Studio | gINT / Strater / RockWorks analogue |
| --- | --- | --- |
| 1 | **Data** — upload Excel workbook | Project database / Excel export |
| 2 | **Validate** — QA metrics, lithology mapping | Data checker, import QA |
| 3 | **Configure** — plan view, hole order, transect A–A′ | Fence line / section definition |
| 4 | **Generate** — SVG profile (fast preview) | Section preview |
| 5 | **Prepare deliverables** — PNG/PDF (one draw); optional Word / batch ZIP | Report sheet / layout export |

> **Note:** Rows 1–4 map to Streamlit’s four workflow steps (Upload → Validate → Configure → Generate). **Prepare deliverables** is a Generate-step export action, not a fifth workflow step.

## UI patterns

- **Compact hero** — after upload, the header shrinks so the figure gets vertical room (reporting focus).
- **Figure-first** — once a profile exists, the SVG sheet stays on top; Validate & Configure move into **Setup — Validate & Configure** (collapsed).
- **Status strip** — shows the figure title, Up to date / Out of date, and the **Generate section** button (`Alt+Shift+G`); use it after changing the figure style or section line.
- **Plan mini-map** — collar scatter in Configure mirrors a plan-view pick for fence orientation.
- **Hole order** — numbered sequence with ↑/↓ matches Strater-style hole ordering for A–A′.
- **Export ribbon** — chips show preset, VE (`auto (fit page)` or the exact value), hole count, transect, up to date / out of date, PNG/PDF readiness.
- **Sidebar sections** — Data, Figure style, Section line, Title block (consulting styles), Export, Advanced. Controls an output style doesn't use are hidden with a "Set by <style>" note.

## Output presets

- **Section sheet (Strater-style)** — hatch legend on chart, ground surface.
- **Consulting report (title block)** — footer title block, groundwater legend; consulting QA often **blocks export on polygon overlaps** by default — disable only after manual review.
- **Chemistry columns (depth + lab values)** — P2-style sticks at exactly 1× VE (locked), red labels by default.
- **Quick preview (chart)** — minimal chart for internal review.
- With a title-block **Map scale** (e.g. `1:1000`) consulting sheets print at exactly that scale when it fits the page; otherwise the sheet is fitted, the SCALE row reads **AS SHOWN** and a section note gives the approximate printed scale (see **Help → Generate and exports**).

## Groundwater and chemistry overlays

- Water table polylines and chemistry “fences” are **schematic connectors** between measured sticks — **not** a potentiometric surface or plume envelope. Figure footers append overlay disclaimers when those layers plot.
- Optional Water column `status`: `measured` (default), `dry`, or `nm` (not measured). Prefer this over omitting rows when a well was visited but dry/NM.
- Chemistry label colours: **All black**, **All red (client P2 style)** (the default for **Chemistry columns**), or **Green / orange / red thresholds**. Limits are **per parameter, in that parameter's own units**: the **Per-parameter limits** editor lists each selected parameter with its unit and a Green ≤ / Orange ≤ pair (chloride-like parameters start on 100 / 250; others on the sheet-wide pair; a workbook `green_max` / `orange_max` on the Environmental sheet seeds the row), and each parameter remembers its limits for the workbook. The sheet-wide **Green ≤ / Orange ≤** pair is the default for parameters without their own row. Reading dots take the same colour as their values, and the threshold key prints below the distance label — one line per plotted parameter when several are shown (`Chloride: green ≤ 100 · orange 100–250 · red > 250 mg/kg` / `Benzene: green ≤ 0.5 · orange 0.5–5 · red > 5 mg/kg`). If a parameter's orange limit is not above its green one there is no orange band: its line reads `green ≤ 50 · red > 50`. **Non-detects** (`<5` in `value`, or `<` / `ND` in the `qualifier` column; the number is the reporting detection limit) print as `<5`, never a bare `5`: in threshold mode a non-detect whose limit is at or below the green limit prints green, and one whose limit is above it prints **grey** (inconclusive — not detected, but the limit is too high to show the guideline is met), never orange or red; the key adds `grey ND (RDL > …)` and the section notes name each case (`BH-3 Benzene at 2 m: detection limit 5 µg/L exceeds guideline 1 µg/L — result inconclusive`). All black / all red just print the `<` text. A workbook `label_color` always wins. The report ZIP's metadata JSON and the Word figure table list the limits used. Turning on **interpolate chemistry between holes** draws depth-matched segments only; keep it off for P2 stick-style figures.
- Chemistry label style (**Label readability over hatched fills**): **Clear strip beside borehole** (default — values sit on a clean white strip right of the column), **Plain coloured text**, **White box**, **Coloured dot, black text**, or **Outlined (white halo) text**. Values stay beside their own borehole in depth order; a value moved off its reading gets a thin leader. Values with no room at the page scale are left off and listed in the section notes (markers stay).
- When water connectors plot, adjacent measured heads may show a schematic **i=Δh/Δx** label. Validate warns if \|i\| is unrealistically large.

### Plan-view flow direction (planar head fit)

- **Validate → Data health details → Groundwater** fits a plane `h = a·E + b·N + c` through **every measured well in the workbook** (not only the section holes) for each water series / `connect_group` nest — the classic three-point problem, extended to least squares when there are more than three wells. Heads use the same datum rules as the section (`btoc` minus stick-up, surveyed holes at true vertical depth); plan position is the collar easting/northing. Dry / NM readings are left out and the last reading per hole is used.
- The table lists the wells used, the flow direction (**toward** the azimuth of decreasing head, degrees clockwise from grid north, e.g. `toward 128° (SE)`), the gradient **i = \|∇h\|**, R² and the largest residual (three wells fit exactly, so R² is shown as *exact*), and how the active section line sits relative to flow: **along section** (< 20°), **oblique**, or **nearly perpendicular** (> 70°). A polyline section uses its overall start → end bearing.
- Warnings: wells nearly in a line (direction cannot be fitted); heads not planar (R² < 0.8 with four or more wells, or a residual > 0.5 m — possible mounding, pumping, or mixed aquifers); flow nearly perpendicular to the section.
- **Apparent gradients:** an on-section **i=Δh/Δx** label is only the component of the true gradient along the line, \|∇h\|·cos θ. On an oblique section it understates the gradient, and on a section across flow it says little about flow at all. With at least three non-collinear wells the section notes (and the PDF QA page) add a line such as *Groundwater flow (planar fit, 5 wells): toward 128° (SE), i = 0.004; section is oblique (42°) — gradients shown are apparent.*
- This is a screening estimate from a planar surface, **not a calibrated flow model** and not a contoured potentiometric map: it assumes one aquifer, a uniform gradient and heads measured at the same time.

## Keyboard shortcuts

- **Alt+Shift+G** (Option+Shift+G on macOS) — Generate section (same as **File → Generate section**). Full list: **Help → Keyboard shortcuts**.

See **Help → Generate and exports** for SVG-first caching, Prepare deliverables, and batch ZIP.

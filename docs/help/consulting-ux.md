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
- **Export ribbon** — chips show preset, VE, hole count, transect, up to date / out of date, PNG/PDF readiness.
- **Sidebar sections** — Data, Figure style, Section line, Title block (consulting styles), Export, Advanced. Controls an output style doesn't use are hidden with a "Set by <style>" note.

## Output presets

- **Section sheet (Strater-style)** — hatch legend on chart, ground surface.
- **Consulting report (title block)** — footer title block, groundwater legend; consulting QA often **blocks export on polygon overlaps** by default — disable only after manual review.
- **Quick preview (chart)** — minimal chart for internal review.

## Groundwater and chemistry overlays

- Water table polylines and chemistry “fences” are **schematic connectors** between measured sticks — **not** a potentiometric surface or plume envelope. Figure footers append overlay disclaimers when those layers plot.
- Optional Water column `status`: `measured` (default), `dry`, or `nm` (not measured). Prefer this over omitting rows when a well was visited but dry/NM.
- Chemistry label colours use Configure green/yellow thresholds (defaults resemble chloride mg/L scales — retune per parameter). Turning on **interpolate chemistry between holes** draws depth-matched segments only; keep it off for P2 stick-style figures.
- When water connectors plot, adjacent measured heads may show a schematic **i=Δh/Δx** label. Validate warns if \|i\| is unrealistically large.

## Keyboard shortcuts

- **Alt+Shift+G** (Option+Shift+G on macOS) — Generate section (same as **File → Generate section**). Full list: **Help → Keyboard shortcuts**.

See **Help → Generate and exports** for SVG-first caching, Prepare deliverables, and batch ZIP.

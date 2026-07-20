# Consulting UX (gINT / Strater / RockWorks workflow)

Cross Section Studio targets **2D fence reporting** — Excel as source of truth, plan view, A–A′ section sheet, then PDF deliverables. It is **not** a 3D modeller (Leapfrog-style).

## Workflow map

| Step | Studio | gINT / Strater / RockWorks analogue |
| --- | --- | --- |
| 1 | **Data source** — upload Excel workbook | Project database / Excel export |
| 2 | **Validate** — QA metrics, lithology mapping | Data checker, import QA |
| 3 | **Configure** — plan view, hole order, transect A–A′ | Fence line / section definition |
| 4 | **Generate** — SVG profile (fast preview) | Section preview |
| 5 | **Prepare PNG & PDF** — raster exports on demand | Report sheet / layout export |

## UI patterns

- **Compact hero** — after upload, the header shrinks so the figure gets vertical room (reporting focus).
- **Figure-first** — once a profile exists, the SVG sheet stays on top; Validate & Configure move into **Setup — Validate & Configure** (collapsed).
- **Regenerate strip** — sticky bar under the hero when a section exists; use **Regenerate** after changing sidebar style or transect.
- **Plan mini-map** — collar scatter in Configure mirrors a plan-view pick for fence orientation.
- **Hole order** — numbered sequence with ↑/↓ matches Strater-style hole ordering for A–A′.
- **Export ribbon** — chips show preset, VE, hole count, transect, fresh/stale, PNG/PDF readiness.
- **Sidebar accordion** — Data source, Section output, Advanced, AI Assist, Consulting report sheet.

## Output presets

- **Section sheet (Strater-style)** — hatch legend on chart, ground surface.
- **Consulting report (title block)** — footer title block, groundwater legend, consulting QA defaults.
- **Quick preview (chart)** — minimal chart for internal review.

## Keyboard shortcuts

- **Ctrl+G** — Generate / Regenerate (same as menubar **File → Generate cross-section**).

See **Help → Generate & exports** for SVG-first caching and PDF preparation.

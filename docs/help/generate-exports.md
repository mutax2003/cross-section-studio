# Generate & exports

Cross Section Studio is **SVG-first**: **Generate Cross-Section** (or `Ctrl+G`) builds the fence diagram as SVG immediately.

## Quick downloads

After Generate:

| Format | When | Use |
|--------|------|-----|
| **SVG** | Immediately after Generate | Review, CAD import, further editing |
| **PNG** | After Prepare | Word / PowerPoint |
| **PDF** | After Prepare | Print / client binders |

## Prepare deliverables

Click **Prepare deliverables (PNG · PDF · Word · package)** once. That builds PNG and PDF in a **single** matplotlib draw, then unlocks the **Drafter package** row:

- **Word figure (.docx)** — PNG + caption/metadata (needs `python-docx`)
- **Copy PNG to clipboard** — browser clipboard (permission-dependent; fall back to Download PNG)
- **Build report ZIP** / **Download report ZIP** — SVG + PNG + PDF + metadata JSON + README (+ Word when available)
- **Save to project folder** — set **Export output folder** under sidebar **Export framing & deliverables**

**Report ZIP** packages the **current** figure. **Multi-transect ZIP** (below) rebuilds a separate figure per Configure batch line.

## Sidebar framing

Under **Export framing & deliverables**: page preset, margins, DPI, fence-only, DRAFT watermark, layer toggles, viewport crop, filename pattern, CAD-friendly SVG layers (Inkscape layer groups when enabled), and optional output folder path.

## Multi-transect ZIP

Configure → **Multi-transect batch ZIP**: one line per transect as `Label | hole1, hole2, …`.

On Generate, **Build multi-transect ZIP** rebuilds each line through the pipeline (distinct figures), then packages them. Check **Include SVG** only when needed (encode is slower). Optional **report_binder.pdf** merges section PDFs when `pypdf` is installed.

Helpers: **Add current transect**, **Fill from recommended** (after Recommended mode has run once), and **Load from workbook Sections** (when the workbook has a **Sections** sheet). An empty batch box is seeded **once** from that sheet on Configure; clearing the box after that does not re-seed — use **Load from workbook Sections** to refresh.

## CAD note

Download SVG for drafting. With **CAD-friendly SVG layers** on, export promotes known groups (`fence`, `tracks`, `water`, `legend`, `surface`, `headers`) to V1 Inkscape layer groups (`inkscape:groupmode="layer"`) and sets Creator metadata to Cross Section Studio CAD. Default SVG stays unchanged when the toggle is off.

## QA before export

If **Block export on polygon overlaps** is on in Configure, resolve overlaps (or clear the gate after manual review) before Generate or batch ZIP. Cosmetic changes (title, VE, hatches, fonts, column width) still need Generate for a new SVG; projection/stratigraphy can reuse cached geometry when only cosmetics change.

## Regenerate

If transect, style, or correlation settings change, **Generate** again. SVG refreshes immediately; run **Prepare deliverables** again for PNG/PDF/Word/ZIP.

# Generate & exports

Cross Section Studio is **SVG-first**: **Generate Cross-Section** builds the fence diagram as SVG immediately.

## Quick downloads

After Generate:

| Format | Use |
|--------|-----|
| **SVG** | Review, CAD import, further editing |
| **PNG** | Word / PowerPoint (after Prepare) |
| **PDF** | Print / client binders (after Prepare) |

## Prepare deliverables

Click **Prepare deliverables (PNG · PDF · Word · package)** once. That builds PNG and PDF in a single matplotlib draw, then unlocks the **Drafter package** row:

- **Word figure (.docx)** — PNG + caption/metadata (needs `python-docx`)
- **Copy PNG** — browser clipboard (permission-dependent; fall back to Download PNG)
- **Report ZIP** — SVG + PNG + PDF + metadata JSON + README (+ Word when available)
- **Save to project folder** — set **Export output folder** under sidebar **Export framing & deliverables**

## Sidebar framing

Under **Export framing & deliverables**: page preset, margins, DPI, fence-only, DRAFT watermark, layer toggles, viewport crop, filename pattern, CAD-friendly SVG tag (metadata hint only — not full CAD layer groups), and optional output folder path.

## Filename-copy ZIP (optional)

Configure → **Batch transect export labels**: one label per line. On Generate, **Build filename-copy ZIP** packages the **current** figure under each filename stem. It does **not** rebuild separate transects.

## CAD note

Download SVG for drafting. The CAD-friendly toggle only adjusts SVG Creator metadata; it does not create AutoCAD/Inkscape layer groups yet.

## Regenerate

If transect, style, or correlation settings change, **Generate** again. SVG refreshes immediately; run **Prepare deliverables** again for PNG/PDF/Word/ZIP.

Cosmetic changes (title, VE, hatches, fonts, column width) still need Generate for a new SVG, but the geometry cache can reuse projection/stratigraphy when only cosmetics change.

# Generate & exports

Cross Section Studio is **SVG-first**: **Generate section** (or `Alt+Shift+G`) builds the fence diagram as SVG immediately.

## Downloads

After Generate, the download buttons sit directly **above the figure**:

| Button | When | Use |
|--------|------|-----|
| **Download PDF · print** | After Prepare | Print / client binders |
| **Download PNG · Word/slides** | After Prepare | Word / PowerPoint |
| **Download SVG · CAD/review** | Immediately after Generate | Review, CAD import, further editing |

Greyed buttons explain why on hover. The green (primary) button is always the next step: **Prepare deliverables** until PNG/PDF exist, then **Download PDF**.

## Prepare deliverables

Click **Prepare deliverables** once. That builds PNG and PDF in a **single** drawing pass (plus the Word file when available) and shows a notice naming the ready files. It then unlocks:

- **Word figure (.docx)** — PNG + caption/metadata (needs `python-docx`)
- **Copy PNG to clipboard** — browser clipboard (permission-dependent; fall back to Download PNG)
- **Prepare report ZIP** / **Download report ZIP** — SVG + PNG + PDF + metadata JSON + README (+ Word when available)
- **Save to project folder** — set **Save exports to folder** (a full path, e.g. `P:\Projects\Job\Figures`) under sidebar **Export**

**Report ZIP** packages the **current** figure. **Batch ZIP** (below) rebuilds a separate figure for each section line in the batch list.

## Sidebar framing

Under sidebar **Export** (margins, crop and CAD layers are under **Advanced**): page preset, margins, DPI, fence-only, DRAFT watermark, layer toggles, viewport crop, filename pattern, CAD-friendly SVG layers (Inkscape layer groups when enabled), and optional output folder path.

## Batch ZIP (several section lines)

Configure → **Several section lines (batch ZIP)**: one section line per row as `Label | hole1, hole2, …`. Each row is checked as you type (for example `C-C': MW-99 not in Collars`).

On Generate, **Prepare batch ZIP** draws each valid line as its own figure and packages them with a README; lines that can't be drawn are listed and left out instead of failing the batch. Tick **Include SVG in batch ZIP** only when needed (SVG is slower). **report_binder.pdf** collects the section PDFs behind a cover page.

Helpers: **Add current section line**, **Fill from suggested lines**, and **Load from workbook Sections** (when the workbook has a **Sections** sheet). An empty batch box is seeded **once** from that sheet on Configure; clearing the box after that does not re-seed — use **Load from workbook Sections** to refresh.

## CAD note

Download SVG for drafting. With **CAD-friendly SVG layers** on, export promotes known groups (`fence`, `tracks`, `water`, `legend`, `surface`, `headers`) to V1 Inkscape layer groups (`inkscape:groupmode="layer"`) and sets Creator metadata to Cross Section Studio CAD. Default SVG stays unchanged when the toggle is off.

## Vertical exaggeration on exports

With **Auto (fit page)** (the default) the figure fills the page and the VE caption shows the VE actually printed (e.g. `VERTICAL EXAGGERATION ≈2.2×`); it is re-measured for the export page preset, so the PNG, PDF and SVG captions match the sheet they are on. Pick a number (1×, 2×, 5×, …) to draw exactly that VE on every page: the plot keeps its proportions and shrinks inside its frame instead of stretching.

## Map scale on consulting sheets

When the title block **Map scale** is set (sidebar or workbook, e.g. `1:1 000`), the consulting sheet draws the section at exactly that horizontal scale on the export page: the plot box is sized to it and centred in its frame, the scale bar reads `SCALE 1:1000`, and the title block SCALE row shows your value. With a chosen VE the box height follows; with Auto VE the plot fills the frame height and the caption shows the measured VE. All horizontal distance on the plot, including the room kept for the last hole's labels, is at that scale.

If the section is too long (or, with a fixed VE, too tall) for the page at that scale, or would fill less than 30 % of the frame width, the sheet is fitted to the frame instead: the SCALE row reads **AS SHOWN**, the scale bar shows the approximate printed scale, and the build reports a QA note such as *Map scale 1:1000 doesn't fit a letter landscape page; printed at approx. 1:1538 (AS SHOWN).* Choose a larger page preset (tabloid) or a smaller scale to print at your value. With Map scale left blank the SCALE row always reads AS SHOWN. A fitted sheet's scale bar always says `APPROX. SCALE` with the measured ratio (e.g. `APPROX. SCALE 1:1280`); only a sheet drawn at its title-block map scale prints an unqualified `SCALE 1:…`.

## QA before export

If **Stop if matched layers overlap** is on in Configure, resolve overlaps (or clear the gate after manual review) before Generate or batch ZIP. Cosmetic changes (title, VE, hatches, fonts, column width) still need Generate for a new SVG; projection/stratigraphy can reuse cached geometry when only cosmetics change.

Water and chemistry connector layers (when plotted) add footer text clarifying they are **schematic** — not potentiometric surfaces or plume contours. Chemistry colour thresholds in Configure are global (not per-parameter guidelines); keep chemistry interpolate off for stick-style P2 figures.

## Generate again

If the section line, style, or layer matching changes, click **Generate section** again. SVG refreshes immediately; run **Prepare deliverables** again for PNG/PDF/Word/ZIP.

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

## File names

Sidebar **Export → File names**:

- **Section title** (default) — e.g. `Site_X_A-A'`. Spaces become `_`, `A - A'` becomes `A-A'`, and the section prime is kept.
- **Project_Figure_Section_Rev** — title-block project and figure numbers, section and revision, e.g. `12345_Fig3_A-A'_Rev_A`.

**Revision / draft tag** is appended (with the DRAFT watermark on, `DRAFT` is added). Names Windows reserves (`CON`, `PRN`, `LPT1` …) get a trailing `_`.

## Batch ZIP (several section lines)

Configure → **Several section lines (batch ZIP)**: one section line per row as `Label | hole1, hole2, …`. Each row is checked as you type (for example `C-C': MW-99 not in Collars`).

On Generate, **Prepare batch ZIP** draws each valid line as its own figure and packages them with a README; lines that can't be drawn are listed and left out instead of failing the batch. Tick **Include SVG in batch ZIP** only when needed (SVG is slower). **report_binder.pdf** collects the section PDFs behind a cover page; every page is fitted (scaled, not cropped) onto one letter landscape sheet size so the binder prints evenly.

Each section is named after **its own** line, not the section on screen: the title's section label is swapped for the line's (title `Site X A-A'` → `Site_X_B-B'.pdf` for line `B-B'`), with the prime kept. The ZIP and binder cover are named for the whole batch, e.g. cover *Site X — Cross Sections A-A', B-B'* and `Site_X_Cross_Sections_A-A'_B-B'.zip`; words after the label stay after it (title `A - A' WITH CHLORIDE` → `B-B'_WITH_CHLORIDE.pdf` and `Cross_Sections_A-A'_B-B'_WITH_CHLORIDE.zip`). Every output style names files the same way. Two lines that would give the same name get `_2`, `_3`.

Helpers: **Add current section line**, **Fill from suggested lines**, and **Load from workbook Sections** (when the workbook has a **Sections** sheet). An empty batch box is seeded **once** from that sheet on Configure; clearing the box after that does not re-seed — use **Load from workbook Sections** to refresh.

## CAD note

Download SVG for drafting. With **CAD-friendly SVG layers** on, export promotes known groups (`fence`, `tracks`, `water`, `legend`, `surface`, `headers`) to V1 Inkscape layer groups (`inkscape:groupmode="layer"`) and sets Creator metadata to Cross Section Studio CAD. Default SVG stays unchanged when the toggle is off.

## Vertical exaggeration on exports

With **Auto (fit page)** (the default) the figure fills the page and the VE caption shows the VE actually printed (e.g. `VERTICAL EXAGGERATION ≈2.2×`); it is re-measured for the export page preset, so the PNG, PDF and SVG captions match the sheet they are on. Pick a number (1×, 2×, 5×, …) to draw exactly that VE on every page: the plot keeps its proportions and shrinks inside its frame instead of stretching.

A workbook **Project** tab `vertical_exaggeration` (a number such as `5`, or `auto`) sets the starting choice when the workbook is uploaded; a new workbook without one, or **Clear workbook**, returns to Auto. A VE you pick yourself stays when you switch **Output style** (it is not reset to the workbook or style default). **Chemistry columns** (P2 style) is always drawn at exactly 1× and its VE control is locked.

## Map scale on consulting sheets

When the title block **Map scale** is set (sidebar or workbook, e.g. `1:1 000`), the consulting sheet draws the section at exactly that horizontal scale on the export page: the plot box is sized to it and centred in its frame, the scale bar reads `SCALE 1:1000`, and the title block SCALE row shows your value. With a chosen VE the box height follows; with Auto VE the plot fills the frame height and the caption shows the measured VE. All horizontal distance on the plot, including the room kept for the last hole's labels, is at that scale.

If the section is too long (or, with a fixed VE, too tall) for the page at that scale, or would fill less than 30 % of the frame width, the sheet is fitted to the frame instead: the SCALE row reads **AS SHOWN**, the scale bar shows the approximate printed scale, and the build reports a QA note such as *Map scale 1:1000 doesn't fit a letter landscape page; printed at approx. 1:1538 (AS SHOWN).* Choose a larger page preset (tabloid) or a smaller scale to print at your value. With Map scale left blank the SCALE row always reads AS SHOWN. A fitted sheet's scale bar always says `APPROX. SCALE` with the measured ratio (e.g. `APPROX. SCALE 1:1280`); only a sheet drawn at its title-block map scale prints an unqualified `SCALE 1:…`.

## Section notes after Generate

Anything you should check before issuing the figure is listed in one amber box under the figure. With **Include QA notes (PDF)** on, the PDF lists them too: on the summary page (letter landscape) for Section sheet and Quick preview, and on an extra letter-landscape **QA notes** page after the sheet for the one-sheet consulting styles (Consulting, Groundwater fence, Chemistry columns, Chemistry + groundwater). That toggle is on by default for Section sheet and off for the consulting styles, whose PDF usually goes to the client; once you set it yourself, your choice is kept across styles.

- **Polygon overlap** — matched layers overlap between two holes.
- **Crossing correlation** — the same layers are logged in opposite order in neighbouring holes (e.g. `Crossing correlation MW-01–MW-02: Sand, Clay logged in opposite order — drawn as pinch-outs`). They are not joined across; with pinch-outs off they are not drawn between those holes. Check `unit_order` or add a Correlations row.
- **Fence clipped** — a fence polygon lost much of its area to a conflicting correlation (`… kept 40% of its area`).
- **Chemistry labels** — values that had no room beside their borehole at this scale are listed by hole (`Chemistry labels: 2 value(s) not printed …`). Their markers are still drawn; a larger page or fewer boreholes per section shows them.
- **Map scale** — the title-block map scale didn't fit the page (see above; the sheet is marked AS SHOWN).

## QA before export

If **Stop if matched layers overlap** is on in Configure, resolve overlaps (or clear the gate after manual review) before Generate or batch ZIP. Cosmetic changes (title, VE, hatches, fonts, column width) still need Generate for a new SVG; projection/stratigraphy can reuse cached geometry when only cosmetics change.

Water and chemistry connector layers (when plotted) add footer text clarifying they are **schematic** — not potentiometric surfaces or plume contours. Chemistry colour thresholds in Configure apply to every plotted parameter (not per-parameter guidelines); keep chemistry interpolate off for stick-style P2 figures.

## Generate again

If the section line, style, or layer matching changes, click **Generate section** again. SVG refreshes immediately; run **Prepare deliverables** again for PNG/PDF/Word/ZIP.

# Cross Section Studio 1.0.0 — release notes

First numbered release after the October 2026 review with the Ecoventure team.
Publish the Windows zip from `scripts/build_windows.ps1` and a
`release-manifest.json` (see `docs/operator-runbook.md`) as GitHub Release assets.

## Figures

- **Lithology legend 261002** is the authoritative colour and hatch scheme (sand
  stipple, silty dashes, "+" loam mixes, cobble circles for gravel). The old Excel
  hex sheet only supplies codes the scheme lacks.
- **Hatch marks print at template density**; legend swatches draw the same marks
  at the same spacing as the section, on every page size.
- **Chemistry values** sit to the right of their own borehole, in depth order,
  never on a column, on a clean strip (or box) so hatching never runs through a
  number. Values that cannot fit are listed in a note instead of being dropped
  silently.
- **Chemistry colour modes:** all black, all red (client P2 style) or
  green / orange / red thresholds, with the key printed under the distance label
  and reading dots coloured like their values. Limits can be set per parameter.
- **Vertical exaggeration:** *Auto (fit page)* fills the page and prints the
  exaggeration actually drawn; choosing 1×, 2×, 5×, 10× or 20× draws exactly
  that. The Chemistry columns style is locked at 1×.
- **Map scale:** a set scale (e.g. 1:200) is drawn exactly when it fits the page;
  otherwise the title block prints AS SHOWN and the scale bar gives the true
  printed scale, with a note saying why.
- **Fence geometry:** pinch-out wedges attach to the hole where the unit was
  logged, neighbouring layers bend to meet them, logging gaps and layers logged in
  opposite order no longer leave white holes, and units found in only one hole are
  drawn.
- **Groundwater:** markers draw on top of the borehole columns; water-level and
  gradient labels keep off lines and each other.
- Complete legends (chemistry sample, ground surface, inferred pinch-out,
  NM / DRY, "Not logged"), readable hole headers at any spacing, and at least a
  quarter-inch printable margin on every page preset.

## Workbooks and checks

- Automatic layer numbering is site-wide, so matching layers line up between
  holes even when a unit pinches out in one of them.
- Logging gaps draw grey as "Not logged" and never lock Generate; other warnings
  need the override tick on consulting styles.
- New data checks: unit_order running against depth, zero-thickness rows,
  coordinates given in degrees, non-detect values in the wrong column.
- Workbooks with a Sections tab seed the batch export; the batch ZIP names each
  file after its own section and prints a cover page for the whole set.

## App and exports

- Sidebar grouped into Data, Figure style, Section line, Title block, Export and
  Advanced, with help on every control.
- Title-block fields survive output-style switches and reset with each workbook.
- Downloads always reflect the latest figure; PDFs are letter size with the QA
  notes on their own page; Word figure packs; optional editable text in SVG for
  moving labels in Illustrator or Inkscape.
- 60-hole sections export in seconds.

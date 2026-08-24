# Generate & exports

Cross Section Studio is **SVG-first**: clicking **Generate Cross-Section** builds the fence diagram immediately as SVG.

## After Generate

- **Download SVG** — ready right away; use for review, CAD, or further editing.
- **Prepare PNG & PDF** — builds both raster deliverables in one matplotlib draw (skipped during Generate to save time).

After Prepare completes, **Download PNG** and **Download PDF** appear on the next run.

## Prepare separately

Open **Prepare formats separately** under the download row if you only need one download button first. Preparing PNG or PDF still builds **both** formats in one step (so the other download is ready without a second redraw).

## Regenerate

If you change transect, style, or correlation settings, click **Generate Cross-Section** again. SVG refreshes immediately; clear or stale PNG/PDF require **Prepare** again after Generate.

Cosmetic style changes (title, VE, hatches, fonts) still require a new Generate for SVG, but the geometry cache can reuse projection/stratigraphy when only those cosmetics change.

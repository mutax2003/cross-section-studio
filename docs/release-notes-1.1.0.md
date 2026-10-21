# Cross Section Studio 1.1.0 — release notes

Hydrogeology release on top of 1.0.0. The Windows zip and `release-manifest.json`
are built and published by the **Release Windows** GitHub Actions workflow
(`.github/workflows/release-windows.yml`).

## Groundwater heads

- **Inclined holes:** water levels, screens and chemistry samples now plot at true
  vertical depth, the same as the lithology.
  - A collar angle with no Deviations rows draws a straight inclined hole.
  - A negative collar angle is read as a mining-style dip (−90 = straight down).
  - A Deviations row at depth 0 sets the collar orientation.
- **Top of casing:** new optional Water column `depth_reference` (`bgs` default, or
  `btoc`). Readings from top of casing subtract the Collars `stick_up_m`.
- **LNAPL (free product):** optional `product_thickness_m` (or `depth_to_product`)
  and `product_density` (SG, default 0.75). The head is corrected to water
  elevation + thickness × SG. The figure shows a hatched product band, a label
  line and a QA note.
- **Monitoring dates:** optional `date` column on Water and Environmental rows.
  - Validate warns when a water series spans more than 7 days.
  - Duplicate readings use the latest date everywhere.
  - Legends and notes name the reading dates.

## Groundwater analysis

- **Flow direction:** a planar head fit over every measured well gives flow
  direction, gradient and fit quality, in Validate and as a figure note.
  - It warns when the section line is oblique or nearly perpendicular to flow,
    because on-section gradients are then apparent.
- **Nested wells:** vertical gradients are computed between screen mid-points
  (positive = downward), and gradient arrows that contradict the heads are flagged.
- **Flowing artesian heads:** drawn with a riser and a legend entry, not reported
  as errors.
- **Water lines:** clipped where they would rise above the ground between wells.
  A nested deep well no longer breaks the shallow line.
- **New Validate checks:**
  - a reading below the bottom of the screen;
  - a water line joining wells screened in different units;
  - a dry upper screen.

## Chemistry

- **Non-detects:** `<5`, `ND (0.5)`, `<RDL 5` and a `qualifier` column are read as
  non-detects and print as "<5 µg/L".
  - In threshold mode, a detection limit above the guideline prints grey and is
    flagged inconclusive.
- **Lab flags:** J, B, E and similar keep the result as detected. U-type flags
  mean non-detect. R and unknown flags are kept, with a warning.
- **Mixed units:** one parameter reported in more than one unit gets a warning.
  Units are never converted silently.

## Figures and exports

- QA notes (LNAPL correction, vertical gradients, water-line clipping, flow) reach
  the PDF QA page in Generate, Prepare deliverables and batch ZIPs.
- Each water series gets its own symbol. "NM" prints only for rows marked `nm`.
  Gradient labels use two significant figures and name the series.

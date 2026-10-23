# Cross Section Studio 1.1.1 — release notes

Chemistry figure follow-ups on top of 1.1.0.

## Chemistry

- **Two-band exceedance colour modes:** Configure's **Parameter label colour**
  now offers **Black / red (exceedance)** and **Green / red (exceedance)** beside
  the all-black, all-red and green / orange / red threshold modes.
  - One limit per parameter (sheet-wide **Exceedance limit ≤**, default 100, and
    a single **Exceedance ≤** column in the per-parameter editor, seeded from the
    workbook `green_max`); values at or below it print black (or green), above
    it red, and the key under the distance label reads e.g.
    `Chloride: black ≤ 100 · red > 100 mg/kg`.
  - Non-detects follow the threshold rule: the at-or-below colour when the
    detection limit is within the limit, grey (inconclusive, with a QA note)
    when it is above. A parameter without a limit prints black and the key says
    `no limit set — black`. A workbook `label_color` still wins.
  - The metadata JSON / Word figure table record the mode and the limits used.
- **Four parameters per section:** the **Parameters to plot** picker stops at
  four (each with its own marker shape); a longer scripted or batch list plots
  its first four in Configure order and a section note names the rest.

## QA / QC

- QA/QC fixes (uncertainty band and figure parity follow-ups).

# About Cross Section Studio

**Cross Section Studio** builds engineering-grade borehole fence diagrams from collar and lithology data. You upload a workbook, choose holes and style, and export vector or raster figures for reports.

## How geology is built

Geometry and geology are **deterministic**: projection, layer correlation, and rendering follow fixed rules in the pipeline. They are not produced by an LLM.

Inter-hole fills are a **rule-based 2D fence diagram** (linear contacts between matched units, midpoint pinch-outs when enabled). This is **not** geostatistical surface interpolation or 3D modelling. Always review Configure correlation preflight and overlap warnings before client delivery.

## Optional AI assist

Free LLM assist can polish **Validate** and **Configure** guidance in the UI only. Assist never replaces the section-building engine and never invents borehole data.

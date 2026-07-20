# Getting started

Cross Section Studio builds borehole fence diagrams from an Excel workbook. **Enter geology in Excel**, then **upload** the file in the app.

## Enter in Excel, then upload

1. **Download template (data entry)** from the welcome card or sidebar — fill **Collars** and **Lithology** (required).
2. **Upload Excel workbook** in the sidebar under Data source (filled template, native workbook, or field export with Lat/Long).
3. Continue with Validate → Configure → Generate.

Geology is **not** edited inside Streamlit widgets; the workbook is the source of truth.

## Workflow

1. **Enter / Upload** — Template + Excel, or sidebar upload of an existing `.xlsx`. Use **Try sample project** (or `Ctrl+Shift+O`) for a ready-made demo.
2. **Validate** — Review parse warnings, lithology codes, and groundwater series. Fix sheet issues before configuring the section.
3. **Configure** — Choose holes, transect order, layout/style, and overlays (water, screens, gradients). Resolve correlation preflight if prompted.
4. **Generate** — Build the cross-section (**SVG is ready immediately**). Click **Prepare PNG & PDF** for deliverable downloads (or prepare one format under the expander). Regenerate after config changes.

## Sample project

On the welcome card, click **Try sample project** (or press `Ctrl+Shift+O`) to load the built-in demo workbook. That path is the fastest way to see Validate → Configure → Generate without preparing your own data.

## Input template

Download **Download template (data entry)** from the welcome card or sidebar. Required tabs are **Collars** and **Lithology**; optional tabs cover Water, Screens, Gradients, Environmental, and project metadata.

For column details and optional sheets, see [Workbook & data entry](workbook-quick.md) and the full [workbook format](../workbook-format.md).

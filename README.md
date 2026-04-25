# Retention Candidate Streamlit App

Run from the repository root:

```powershell
streamlit run web_float\app.py
```

For Streamlit Community Cloud, set the entrypoint to:

```text
web_float/app.py
```

`web_float/requirements.txt` sits next to the entrypoint, which Streamlit Community Cloud can use for dependency installation. If you want the custom theme online, copy `web_float/.streamlit/config.toml` to `.streamlit/config.toml` at the repository root before deploying.

Sentinel Hub credentials should be configured as Streamlit secrets or environment variables:

```toml
SH_CLIENT_ID = "..."
SH_CLIENT_SECRET = "..."
```

The app computes a new AOI from a bounding box supplied in the sidebar. By default, the flood mask uses DEM-driven synthetic SAR. If enabled in the sidebar, the app can request live Sentinel-1 IW VV/VH dry and wet patches through Sentinel Hub and use those scenes for the flood mask.

Default Budapest bounding box:

```text
18.529816,47.303913,19.618149,47.689428
```

Magdala bounding box:

```text
11.413078,50.893966,11.481099,50.916455
```

When you click **COMPUTE**, candidates, GeoJSON, metadata, and map PNGs are kept in Streamlit session memory. The deployed app does not write generated AOI outputs, uploaded DEM files, downloaded DEM files, CSVs, GeoJSON, metadata, or map images to disk.

The in-memory metadata stores the flood data source and two demonstrative flood-fill estimates. The rainfall-only estimate uses the flood mask, an approximate storage depth up to the 90th percentile flood-mask elevation, and the sidebar rainfall/runoff assumptions. The routed estimate adds DEM connectivity, the strongest flow-accumulation inlet, a triangular river-discharge hydrograph, and a captured-flow fraction. The discharge inputs can be set from a GloFAS nearest-cell summary, but the app does not yet download GloFAS data automatically. Both estimates are for comparison on the map, not a calibrated 2D hydrodynamic forecast.

The terrain overlay uses Leaflet inside Streamlit with OpenTopoMap as the default basemap.

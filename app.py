from __future__ import annotations

import base64
import html
import hashlib
import json
import math
import os
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from pipeline import compute_candidates, estimate_area_km2, estimate_width_height_km, parse_bounds


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "outputs"
DEFAULT_DEM_PATH = DEFAULT_OUTPUT_DIR / "copernicus_dem_glo30.tif"
AUGSBURG_BOUNDS = [10.523529, 48.336625, 10.964355, 48.563431]
BUDAPEST_BOUNDS = [19.000000, 47.450000, 19.120000, 47.540000]
BRUSSELS_BOUNDS = [4.233856, 50.788792, 4.505939, 50.878886]
GENOA_BOUNDS = [8.869705, 44.386882, 9.005747, 44.437831]
MAGDALA_BOUNDS = [11.413078, 50.893966, 11.481099, 50.916455]
MARSEILLE_BOUNDS = [5.339484, 43.273018, 5.407505, 43.298979]
ZURICH_BOUNDS = [8.505135, 47.358275, 8.573155, 47.382428]
SELECT_LOCATION = "Select a location"
PRESETS = {
    "Augsburg": AUGSBURG_BOUNDS,
    "Budapest": BUDAPEST_BOUNDS,
    "Brussels": BRUSSELS_BOUNDS,
    "Genoa": GENOA_BOUNDS,
    "Magdala": MAGDALA_BOUNDS,
    "Marseille": MARSEILLE_BOUNDS,
    "Zurich": ZURICH_BOUNDS,
    "Custom": BUDAPEST_BOUNDS,
}
PRESET_OPTIONS = [SELECT_LOCATION, *PRESETS.keys()]
FEATURE_COLS = ["score", "slope", "twi", "tpi", "flow_accum"]
BEST_CANDIDATE_STAR_COUNT = 5
FALLBACK_DEM_SOURCE = "Synthetic fallback DEM"
PRODUCTION_DEM_SOURCE = "Copernicus DEM GLO-30"
DEM_CAUTION = (
    "For production scoring, use pre-calculated Copernicus DEM GLO-30 derivatives. "
    "Do not treat a DEM calculated from Sentinel-1 interferometry as production-grade "
    "without validation; vegetation, weather, atmosphere, temporal decorrelation, and "
    "phase errors can make it noisy."
)


st.set_page_config(
    page_title="Water Stellar",
    layout="wide",
    initial_sidebar_state="expanded",
)


def resolve_dem_path(raw_path: str) -> Path | None:
    cleaned = raw_path.strip().strip('"').strip("'")
    if not cleaned:
        return None
    path = Path(cleaned).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path


def load_local_env(path: Path = ROOT / ".env") -> None:
    try:
        for key in ("SH_CLIENT_ID", "SH_CLIENT_SECRET", "COPERNICUS_DEM_PATH"):
            value = st.secrets.get(key)
            if value:
                os.environ[key] = str(value)
    except Exception:
        pass

    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key in {"SH_CLIENT_ID", "SH_CLIENT_SECRET"}:
            os.environ[key] = value
        else:
            os.environ.setdefault(key, value)


def has_sentinelhub_credentials() -> bool:
    load_local_env()
    return bool((os.getenv("SH_CLIENT_ID") or "").strip() and (os.getenv("SH_CLIENT_SECRET") or "").strip())


def estimate_dem_pixels(bounds: list[float], resolution_m: float) -> int:
    width_km, height_km = estimate_width_height_km(bounds)
    return math.ceil(width_km * 1000 / resolution_m) * math.ceil(height_km * 1000 / resolution_m)


@st.cache_data
def image_data_uri(image_bytes: bytes) -> str:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def viridis_hex(value: float, vmin: float, vmax: float) -> str:
    stops = [
        (68, 1, 84),
        (59, 82, 139),
        (33, 145, 140),
        (94, 201, 98),
        (253, 231, 37),
    ]
    t = 0.5 if vmax == vmin else max(0.0, min(1.0, (value - vmin) / (vmax - vmin)))
    scaled = t * (len(stops) - 1)
    idx = min(len(stops) - 2, int(scaled))
    local = scaled - idx
    rgb = tuple(round(stops[idx][i] + (stops[idx + 1][i] - stops[idx][i]) * local) for i in range(3))
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def tile_sources() -> dict:
    return {
        "Terrain": {
            "url": "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
            "attribution": "Map data: OpenStreetMap contributors | Map style: CartoDB Positron",
            "max_zoom": 20,
        },
        "Physical": {
            "url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
            "attribution": "Tiles: Esri World Topographic Map",
            "max_zoom": 19,
        },
        "Street": {
            "url": "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
            "attribution": "Map data: OpenStreetMap contributors",
            "max_zoom": 19,
        },
    }


def raw_map_html(tile_name: str, map_id: str) -> str:
    tile = tile_sources()[tile_name]
    payload = {"tile": tile}
    return f"""
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <style>
      html, body, #{map_id} {{ height: 100%; margin: 0; }}
      .leaflet-container {{ font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    </style>
  </head>
  <body>
    <div id="{map_id}"></div>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script>
      const payload = {json.dumps(payload)};
      const map = L.map("{map_id}", {{ preferCanvas: true, scrollWheelZoom: true }}).setView([20, 0], 2);
      L.tileLayer(payload.tile.url, {{
        maxZoom: payload.tile.max_zoom,
        attribution: payload.tile.attribution
      }}).addTo(map);
    </script>
  </body>
</html>
"""


def leaflet_map_html(
    candidates: pd.DataFrame,
    geojson: dict,
    bounds: list[float],
    highlight_count: int,
    tile_name: str,
    map_id: str,
    flood_overlay_uri: str | None,
    flood_fill: dict | None,
    routed_flood_fill: dict | None,
) -> str:
    tile = tile_sources()[tile_name]
    score_min = float(candidates["score"].min())
    score_max = float(candidates["score"].max())
    features = sorted(geojson["features"], key=lambda item: item["properties"]["score"], reverse=True)
    markers = []

    for rank, feature in enumerate(features, start=1):
        lon, lat = feature["geometry"]["coordinates"]
        props = feature["properties"]
        color = viridis_hex(float(props["score"]), score_min, score_max)
        is_star = rank <= BEST_CANDIDATE_STAR_COUNT
        radius = 8 if rank <= highlight_count else 4
        stroke = "#101816" if rank <= highlight_count else "#ffffff"
        weight = 2 if rank <= highlight_count else 1
        popup = (
            f"<b>Rank {rank}</b><br>"
            f"Score: {props['score']:.3f}<br>"
            f"Lon/Lat: {lon:.4f}, {lat:.4f}<br>"
            f"Flood frequency: {props['flood_frequency']:.2f}<br>"
            f"Slope: {props['slope']:.2f}<br>"
            f"TWI: {props['twi']:.2f}<br>"
            f"TPI: {props['tpi']:.2f}<br>"
            f"Flow accumulation: {props['flow_accum']:.1f}"
        )
        markers.append(
            {
                "rank": rank,
                "lat": lat,
                "lon": lon,
                "color": color,
                "is_star": is_star,
                "radius": radius,
                "stroke": stroke,
                "weight": weight,
                "popup": popup,
                "tooltip": f"Rank {rank} | score {props['score']:.3f}",
            }
        )

    map_payload = {
        "tile": tile,
        "markers": markers,
        "bounds": [[bounds[1], bounds[0]], [bounds[3], bounds[2]]],
        "flood_overlay_uri": flood_overlay_uri,
        "flood_fill": flood_fill or {},
        "routed_flood_fill": routed_flood_fill or {},
    }

    return f"""
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <style>
      html, body, #{map_id} {{ height: 100%; margin: 0; }}
      .leaflet-container {{ font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
      .legend {{
        background: white;
        border: 1px solid #cfd8d2;
        border-radius: 6px;
        box-shadow: 0 8px 18px rgba(0,0,0,.16);
        color: #1f2926;
        line-height: 1.35;
        padding: 10px 12px;
      }}
      .legend-bar {{
        width: 150px;
        height: 10px;
        margin: 7px 0 4px;
        border-radius: 999px;
        background: linear-gradient(90deg, #440154, #3b528b, #21918c, #5ec962, #fde725);
      }}
      .legend-scale {{ display: flex; justify-content: space-between; gap: 18px; color: #65716b; font-size: 12px; }}
      .best-candidate-star {{
        color: #f5b301;
        font-size: 25px;
        font-weight: 900;
        line-height: 25px;
        text-align: center;
        text-shadow:
          -1px -1px 0 #111816,
           1px -1px 0 #111816,
          -1px  1px 0 #111816,
           1px  1px 0 #111816,
           0 2px 5px rgba(0,0,0,.35);
      }}
      .map-metrics {{
        background: white;
        border: 1px solid #cfd8d2;
        border-radius: 6px;
        box-shadow: 0 8px 18px rgba(0,0,0,.16);
        color: #1f2926;
        min-width: 190px;
        padding: 10px 12px;
      }}
      .map-metrics strong {{ display: block; font-size: 13px; margin-bottom: 6px; }}
      .map-metrics-row {{ display: flex; justify-content: space-between; gap: 14px; font-size: 12px; line-height: 1.55; }}
      .map-metrics-value {{ font-weight: 700; text-align: right; }}
      .map-metrics-note {{ color: #65716b; font-size: 11px; line-height: 1.25; margin-top: 6px; }}
    </style>
  </head>
  <body>
    <div id="{map_id}"></div>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script>
      const payload = {json.dumps(map_payload)};
      const formatNumber = (value, digits = 1) => Number.isFinite(value) ? value.toFixed(digits) : "n/a";
      const map = L.map("{map_id}", {{ preferCanvas: true, scrollWheelZoom: true }});
      L.tileLayer(payload.tile.url, {{
        maxZoom: payload.tile.max_zoom,
        attribution: payload.tile.attribution
      }}).addTo(map);

      const aoi = L.rectangle(payload.bounds, {{
        color: "#111816",
        weight: 2,
        fill: false,
        dashArray: "6 5"
      }}).addTo(map);

      if (payload.flood_overlay_uri) {{
        L.imageOverlay(payload.flood_overlay_uri, payload.bounds, {{
          opacity: 0.28,
          interactive: false
        }}).addTo(map);
      }}

      payload.markers.forEach((marker) => {{
        if (marker.is_star) {{
          L.marker([marker.lat, marker.lon], {{
            icon: L.divIcon({{
              className: "best-candidate-star",
              html: "★",
              iconSize: [25, 25],
              iconAnchor: [12, 12],
              popupAnchor: [0, -12],
              tooltipAnchor: [13, 0]
            }})
          }})
            .bindPopup(marker.popup)
            .bindTooltip(marker.tooltip)
            .addTo(map);
        }} else {{
          L.circleMarker([marker.lat, marker.lon], {{
            radius: marker.radius,
            color: marker.stroke,
            weight: marker.weight,
            fillColor: marker.color,
            fillOpacity: 0.86
          }})
            .bindPopup(marker.popup)
            .bindTooltip(marker.tooltip)
            .addTo(map);
        }}
      }});

      map.fitBounds(payload.bounds, {{ padding: [22, 22] }});

      const legend = L.control({{ position: "bottomright" }});
      legend.onAdd = function() {{
        const div = L.DomUtil.create("div", "legend");
        div.innerHTML = `
          <strong>Suitability score</strong>
          <div class="legend-bar"></div>
          <div class="legend-scale"><span>{score_min:.3f}</span><span>{score_max:.3f}</span></div>
          <div style="margin-top:7px;color:#1f2926;font-size:12px;">★ Top {BEST_CANDIDATE_STAR_COUNT} candidates</div>
          <div style="margin-top:7px;color:#2563eb;font-size:12px;">Blue overlay: flood mask</div>
          <div style="margin-top:7px;color:#65716b;font-size:12px;">Dashed box: AOI bounds</div>
        `;
        return div;
      }};
      legend.addTo(map);

      if (payload.flood_fill && Object.keys(payload.flood_fill).length) {{
        const metrics = L.control({{ position: "topright" }});
        metrics.onAdd = function() {{
          const data = payload.flood_fill;
          const routed = payload.routed_flood_fill || {{}};
          const div = L.DomUtil.create("div", "map-metrics");
          const fillHours = Number.isFinite(data.fill_hours) ? formatNumber(data.fill_hours, 1) + " h" : "n/a";
          const routedHours = Number.isFinite(routed.fill_hours)
            ? formatNumber(routed.fill_hours, 1) + " h"
            : (Number.isFinite(routed.hydrograph_duration_h) ? ">" + formatNumber(routed.hydrograph_duration_h, 0) + " h" : "n/a");
          div.innerHTML = `
            <strong>Flood fill estimate</strong>
            <div class="map-metrics-row"><span>Fill time</span><span class="map-metrics-value">${{fillHours}}</span></div>
            <div class="map-metrics-row"><span>Routed fill</span><span class="map-metrics-value">${{routedHours}}</span></div>
            <div class="map-metrics-row"><span>Flood area</span><span class="map-metrics-value">${{formatNumber(data.flood_area_km2, 2)}} km2</span></div>
            <div class="map-metrics-row"><span>Storage</span><span class="map-metrics-value">${{formatNumber((data.storage_volume_m3 || 0) / 1000, 0)}}k m3</span></div>
            <div class="map-metrics-row"><span>Peak Q</span><span class="map-metrics-value">${{formatNumber(routed.peak_discharge_m3_s, 0)}} m3/s</span></div>
            <div class="map-metrics-note">Routed estimate uses DEM connectivity and a river-discharge hydrograph.</div>
          `;
          L.DomEvent.disableClickPropagation(div);
          return div;
        }};
        metrics.addTo(map);
      }}
    </script>
  </body>
</html>
"""


def format_hours(value: float | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return "n/a"
    if value < 48:
        return f"{value:.1f} h"
    return f"{value / 24:.1f} d"


def format_routed_hours(routed_fill: dict | None) -> str:
    if not routed_fill:
        return "n/a"
    value = routed_fill.get("fill_hours")
    if value is not None and math.isfinite(float(value)):
        return format_hours(float(value))
    duration = routed_fill.get("hydrograph_duration_h")
    if duration is not None:
        return f">{duration:.0f} h"
    return "not filled"


def routed_fill_fraction(routed_fill: dict | None) -> float | None:
    if not routed_fill:
        return None
    value = routed_fill.get("fill_fraction")
    if value is not None and math.isfinite(float(value)):
        return max(0.0, min(1.0, float(value)))

    storage = routed_fill.get("routed_storage_volume_m3")
    duration = routed_fill.get("hydrograph_duration_h")
    local_runoff = routed_fill.get("local_runoff_m3_s", 0.0)
    capture = routed_fill.get("effective_capture_fraction", routed_fill.get("capture_fraction", 0.0))
    base_q = routed_fill.get("base_discharge_m3_s")
    peak_q = routed_fill.get("peak_discharge_m3_s")
    if not storage or not duration or base_q is None or peak_q is None:
        return None
    average_q = ((float(base_q) + float(peak_q)) / 2) * float(capture) + float(local_runoff)
    delivered_m3 = average_q * float(duration) * 3600
    return max(0.0, min(1.0, delivered_m3 / float(storage)))


def render_metric_cards(candidates: pd.DataFrame, bounds: list[float], metadata: dict | None = None) -> None:
    area = estimate_area_km2(bounds)
    width_km, height_km = estimate_width_height_km(bounds)
    top_score = candidates["score"].max()
    mean_score = candidates["score"].mean()
    flood_fill = (metadata or {}).get("flood_fill", {})
    routed_fill = (metadata or {}).get("routed_flood_fill", {})
    cols = st.columns(6)
    cols[0].metric("Candidates", f"{len(candidates):,}")
    cols[1].metric("Top score", f"{top_score:.3f}")
    cols[2].metric("Mean score", f"{mean_score:.3f}")
    cols[3].metric("Rain fill", format_hours(flood_fill.get("fill_hours")))
    cols[4].metric("Routed fill", format_routed_hours(routed_fill))
    cols[5].metric("AOI", f"{area:,.0f} km2", f"{width_km:.1f} x {height_km:.1f} km")


def current_result() -> tuple[pd.DataFrame, dict, dict, dict[str, bytes], list[float], str, str] | None:
    state = st.session_state.get("computed_aoi")
    if not state:
        return None
    return (
        state["candidates"],
        state["geojson"],
        state["metadata"],
        state["map_images"],
        state["bounds"],
        state["label"],
        state["run_id"],
    )


def main() -> None:
    st.markdown("### Water Stellar")
    st.caption("Streamlit visualization of top retention sites with basemap overlay and offline AOI scoring.")

    with st.sidebar:
        st.markdown("### Water Stellar")
        st.divider()
        st.header("AOI Controls")
        preset = st.selectbox("Preset", PRESET_OPTIONS, index=0)
        location_selected = preset != SELECT_LOCATION
        default_bounds = PRESETS.get(preset, BUDAPEST_BOUNDS)
        bbox_text = st.text_input(
            "Bounding box",
            value="" if not location_selected else ",".join(f"{value:.6f}" for value in default_bounds),
            help="Format: min_lon,min_lat,max_lon,max_lat",
            disabled=not location_selected,
        )
        aoi_name = st.text_input(
            "Output folder label",
            value="" if not location_selected else preset.lower().replace(" ", "_"),
            disabled=not location_selected,
        )
        resolution = st.select_slider("Analysis grid", options=[160, 220, 300, 420], value=300)
        top_n = st.slider("Candidate count", 50, 500, 250, step=50)
        st.divider()
        st.header("DEM Source")
        download_dem = st.checkbox("Download Copernicus DEM for selected AOI", value=True)
        dem_resolution_m = st.select_slider("DEM download resolution", options=[30, 60, 90], value=30)
        dem_path_text = st.text_input(
            "Existing Copernicus DEM GeoTIFF",
            value=str(DEFAULT_DEM_PATH),
            help="Optional local Copernicus DEM GLO-30 GeoTIFF. If download is enabled, this is used only when the file already exists.",
        )
        uploaded_dem = st.file_uploader("Upload DEM GeoTIFF", type=["tif", "tiff"])
        uploaded_dem_bytes = uploaded_dem.getvalue() if uploaded_dem is not None else None
        if uploaded_dem is not None:
            st.caption(f"Ready: uploaded DEM `{uploaded_dem.name}`")
        require_real_dem = st.checkbox("Require real DEM", value=True)
        dem_path_preview = resolve_dem_path(dem_path_text)
        dem_ready = bool(dem_path_preview and dem_path_preview.exists())
        credentials_ready = has_sentinelhub_credentials()
        if uploaded_dem_bytes is None and download_dem and credentials_ready:
            st.caption("Ready to download a fresh Copernicus DEM GLO-30 GeoTIFF for the selected AOI on compute.")
        elif uploaded_dem_bytes is None and download_dem:
            st.error("DEM download is enabled, but SH_CLIENT_ID and SH_CLIENT_SECRET are not available in Streamlit secrets, environment variables, or `.env`.")
        elif uploaded_dem_bytes is None and dem_ready:
            st.caption(f"Ready: `{dem_path_preview}`")
        elif uploaded_dem_bytes is None and require_real_dem:
            st.error(
                "Real DEM is required, but the configured GeoTIFF path does not exist. "
                f"Checked: `{dem_path_preview or 'blank path'}`"
            )
        elif uploaded_dem_bytes is None:
            st.caption(f"Current app scoring will use {FALLBACK_DEM_SOURCE} if no GeoTIFF is found.")
        st.warning(DEM_CAUTION)
        st.divider()
        st.header("Flood Data")
        use_live_sentinel1 = st.checkbox("Use live Sentinel-1 SAR for flood mask", value=False)
        sentinel1_dry_start = st.text_input("Sentinel-1 dry start", value="2023-01-01", disabled=not use_live_sentinel1)
        sentinel1_dry_end = st.text_input("Sentinel-1 dry end", value="2023-01-31", disabled=not use_live_sentinel1)
        sentinel1_wet_start = st.text_input("Sentinel-1 wet start", value="2023-04-01", disabled=not use_live_sentinel1)
        sentinel1_wet_end = st.text_input("Sentinel-1 wet end", value="2023-04-30", disabled=not use_live_sentinel1)
        if use_live_sentinel1 and credentials_ready:
            st.caption("Ready to request Sentinel-1 IW VV/VH dry and wet patches through Sentinel Hub.")
        elif use_live_sentinel1:
            st.error("Sentinel-1 mode requires SH_CLIENT_ID and SH_CLIENT_SECRET in Streamlit secrets, environment variables, or `.env`.")
        else:
            st.caption("Current flood mask mode uses DEM-driven synthetic SAR.")
        st.divider()
        st.header("Fill Estimate")
        rainfall_intensity = st.slider("Rainfall intensity (mm/h)", 5, 100, 25, step=5)
        runoff_coefficient = st.slider("Runoff coefficient", 0.10, 1.00, 0.55, step=0.05)
        st.caption("Rainfall-only fill time is a direct-rainfall storage calculation from the DEM flood mask.")
        st.subheader("Routed Estimate")
        routed_base_discharge = st.number_input("Base river discharge (m3/s)", min_value=0.0, max_value=10000.0, value=20.0, step=5.0)
        routed_peak_discharge = st.number_input("Peak river discharge (m3/s)", min_value=0.0, max_value=100000.0, value=120.0, step=10.0)
        routed_duration = st.slider("Hydrograph duration (hours)", 6, 720, 72, step=6)
        routed_capture_fraction = st.slider("Captured river flow fraction", 0.00, 1.00, 0.08, step=0.01)
        st.caption("Routed fill time uses DEM connectivity and a triangular discharge hydrograph. These fields can represent a GloFAS nearest-cell hydrograph summary.")

        if location_selected:
            try:
                requested_bounds = parse_bounds(bbox_text)
                width_km, height_km = estimate_width_height_km(requested_bounds)
                dem_pixels = estimate_dem_pixels(requested_bounds, dem_resolution_m)
                st.caption(
                    f"Area: {estimate_area_km2(requested_bounds):,.1f} km2 | "
                    f"{width_km:.1f} x {height_km:.1f} km | DEM request: {dem_pixels:,} pixels"
                )
                bounds_valid = True
                if download_dem and dem_pixels > 2_500_000:
                    st.error("Selected AOI is too large for one DEM request. Use a smaller AOI or 60/90 m DEM resolution.")
                    bounds_valid = False
            except ValueError as exc:
                st.error(str(exc))
                requested_bounds = default_bounds
                bounds_valid = False
        else:
            requested_bounds = None
            bounds_valid = False
            st.session_state.pop("computed_aoi", None)
            st.session_state.pop("computed_signature", None)

        real_dem_available = (uploaded_dem_bytes is not None) or (download_dem and credentials_ready) or ((not download_dem) and dem_ready)
        compute_disabled = (not bounds_valid) or (require_real_dem and not real_dem_available) or (use_live_sentinel1 and not credentials_ready)
        if uploaded_dem_bytes is not None:
            dem_path = None
        elif download_dem:
            dem_path = None
        else:
            dem_path = str(dem_path_preview) if dem_ready else None

        def run_compute() -> None:
            if requested_bounds is None:
                return
            compute_name = aoi_name.strip() or preset.lower().replace(" ", "_")
            signature = {
                "preset": preset,
                "bounds": requested_bounds,
                "name": compute_name,
                "resolution": resolution,
                "top_n": top_n,
                "dem_path": dem_path,
                "dem_upload_name": uploaded_dem.name if uploaded_dem is not None else None,
                "require_real_dem": require_real_dem,
                "download_dem": download_dem,
                "dem_resolution_m": dem_resolution_m,
                "rainfall_intensity_mm_h": rainfall_intensity,
                "runoff_coefficient": runoff_coefficient,
                "routed_base_discharge_m3_s": routed_base_discharge,
                "routed_peak_discharge_m3_s": routed_peak_discharge,
                "routed_hydrograph_duration_h": routed_duration,
                "routed_capture_fraction": routed_capture_fraction,
                "use_live_sentinel1": use_live_sentinel1,
                "sentinel1_dry_start": sentinel1_dry_start,
                "sentinel1_dry_end": sentinel1_dry_end,
                "sentinel1_wet_start": sentinel1_wet_start,
                "sentinel1_wet_end": sentinel1_wet_end,
            }
            with st.spinner("Computing flood, terrain, and suitability layers..."):
                result = compute_candidates(
                    bounds=requested_bounds,
                    name=compute_name,
                    height=resolution,
                    width=resolution,
                    top_n=top_n,
                    dem_path=dem_path,
                    dem_bytes=uploaded_dem_bytes,
                    require_real_dem=require_real_dem,
                    download_dem=download_dem,
                    dem_resolution_m=dem_resolution_m,
                    rainfall_intensity_mm_h=rainfall_intensity,
                    runoff_coefficient=runoff_coefficient,
                    routed_base_discharge_m3_s=routed_base_discharge,
                    routed_peak_discharge_m3_s=routed_peak_discharge,
                    routed_hydrograph_duration_h=routed_duration,
                    routed_capture_fraction=routed_capture_fraction,
                    use_live_sentinel1=use_live_sentinel1,
                    sentinel1_dry_start=sentinel1_dry_start,
                    sentinel1_dry_end=sentinel1_dry_end,
                    sentinel1_wet_start=sentinel1_wet_start,
                    sentinel1_wet_end=sentinel1_wet_end,
                )
            st.session_state["computed_aoi"] = {
                "run_id": result["run_id"],
                "candidates": result["candidates"],
                "geojson": result["geojson"],
                "metadata": result["metadata"],
                "map_images": result["map_images"],
                "bounds": requested_bounds,
                "label": compute_name,
            }
            st.session_state["computed_signature"] = signature
            st.cache_data.clear()
            st.success("Computed candidates in memory.")

        if st.button("COMPUTE", type="primary", disabled=compute_disabled):
            run_compute()
            st.rerun()

        st.divider()
        st.header("Map Controls")
        tile_name = st.radio("Basemap", ["Terrain", "Physical", "Street"], horizontal=False)
        show_flood_overlay = st.checkbox("Show flood zone overlay", value=True)
        highlight_count = st.slider("Highlight top candidates", 5, 500, 10, step=5)
        table_count = st.slider("Rows in ranked table", 10, 100, 25, step=5)
        st.divider()
        st.caption(
            "AOI scoring can use either DEM-driven synthetic SAR or live Sentinel-1 SAR for the flood mask. "
            "Terrain derivatives come from Copernicus DEM GLO-30 when a real DEM is supplied or downloaded."
        )
        st.caption(
            "Basemaps are visual map tiles only; "
            "they are not the analytical DEM used by the suitability model."
        )

    result = current_result()
    if result is None:
        st.info("Select a location to compute candidates with the configured real DEM GeoTIFF.")
        components.html(
            raw_map_html(tile_name=tile_name, map_id="raw_start_map"),
            height=680,
            scrolling=False,
        )
        return

    candidates, geojson, metadata, map_images, bounds, label, run_id = result
    candidates = candidates.sort_values("score", ascending=False).reset_index(drop=True)
    if "rank" not in candidates.columns:
        candidates.insert(0, "rank", candidates.index + 1)
    highlight_count = min(highlight_count, len(candidates))
    table_count = min(table_count, len(candidates))
    active_dem_source = metadata.get("dem_source", FALLBACK_DEM_SOURCE)
    active_is_real_dem = bool(metadata.get("is_real_dem", False))
    active_flood_source = metadata.get("flood_data_source", "DEM-driven synthetic SAR")

    st.info(f"Active AOI: {label} | `{run_id}`")
    st.caption(
        f"Analysis DEM source: {active_dem_source}. "
        f"Production source: {PRODUCTION_DEM_SOURCE}."
    )
    st.caption(f"Flood mask source: {active_flood_source}.")
    if active_is_real_dem:
        st.success("This AOI was computed with a real DEM source.")
    else:
        st.warning(f"This AOI was computed with {FALLBACK_DEM_SOURCE}. Recompute with a Copernicus DEM GeoTIFF for real terrain derivatives.")
        st.warning(DEM_CAUTION)
    st.caption("Computed AOIs are kept in memory for this Streamlit session.")
    render_metric_cards(candidates, bounds, metadata)

    st.subheader("Candidate Map Overlay")
    lon_min, lon_max = candidates["lon"].min(), candidates["lon"].max()
    lat_min, lat_max = candidates["lat"].min(), candidates["lat"].max()
    st.caption(
        f"Active bounds: `{bounds}` | Candidate extent: "
        f"lon {lon_min:.6f} to {lon_max:.6f}, lat {lat_min:.6f} to {lat_max:.6f}"
    )
    st.caption("Candidate points are read from GeoJSON and overlaid on visual basemap tiles. Click a point for details.")
    flood_overlay_bytes = map_images.get("flood_zones_overlay.png")
    flood_overlay_uri = image_data_uri(flood_overlay_bytes) if show_flood_overlay and flood_overlay_bytes else None
    flood_fill = metadata.get("flood_fill")
    routed_flood_fill = metadata.get("routed_flood_fill")
    if flood_fill:
        st.caption(
            "Rainfall-only fill estimate: "
            f"{format_hours(flood_fill.get('fill_hours'))} at "
            f"{flood_fill.get('rainfall_intensity_mm_h', 0):.0f} mm/h rainfall and "
            f"{flood_fill.get('runoff_coefficient', 0):.2f} runoff coefficient. "
            "This is a heuristic storage estimate."
        )
    else:
        st.warning("Flood fill estimate is not available for this AOI. Recompute it with the current pipeline.")
    if routed_flood_fill:
        routed_fraction = routed_fill_fraction(routed_flood_fill)
        fraction_text = f", filling about {routed_fraction:.0%} of routed storage" if routed_fraction is not None else ""
        st.caption(
            "Routed fill estimate: "
            f"{format_routed_hours(routed_flood_fill)} using peak discharge "
            f"{routed_flood_fill.get('peak_discharge_m3_s', 0):.0f} m3/s, "
            f"{routed_flood_fill.get('effective_capture_fraction', 0):.3f} effective capture fraction, and "
            f"{routed_flood_fill.get('connected_fraction', 0):.0%} connected flood-mask storage"
            f"{fraction_text}."
        )
    else:
        st.warning("Routed fill estimate is not available for this AOI. Recompute it with the current pipeline.")
    if show_flood_overlay and not flood_overlay_bytes:
        st.warning("Map overlay is not available for this computed AOI.")
    map_digest = hashlib.sha1(
        f"{run_id}|{tile_name}|{highlight_count}|{show_flood_overlay}|{bounds}|{flood_fill}|{routed_flood_fill}".encode("utf-8")
    ).hexdigest()[:10]
    map_id = f"map_{map_digest}"
    components.html(
        leaflet_map_html(
            candidates,
            geojson,
            bounds=bounds,
            highlight_count=highlight_count,
            tile_name=tile_name,
            map_id=map_id,
            flood_overlay_uri=flood_overlay_uri,
            flood_fill=flood_fill,
            routed_flood_fill=routed_flood_fill,
        ),
        height=680,
        scrolling=False,
    )

    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("Top Candidates")
        display_cols = ["rank", "lon", "lat", "score", "flood_frequency", "slope", "twi", "tpi", "flow_accum"]
        st.dataframe(
            candidates.loc[: table_count - 1, display_cols],
            hide_index=True,
            use_container_width=True,
            column_config={
                "score": st.column_config.NumberColumn(format="%.3f"),
                "lon": st.column_config.NumberColumn(format="%.4f"),
                "lat": st.column_config.NumberColumn(format="%.4f"),
                "flow_accum": st.column_config.NumberColumn(format="%.1f"),
            },
        )

    with right:
        st.subheader("Score Distribution")
        st.bar_chart(candidates["score"].round(3).value_counts().sort_index(), height=320)

    st.subheader("Feature Correlation")
    corr = candidates[FEATURE_COLS].corr(numeric_only=True)
    st.dataframe(
        corr.style.background_gradient(cmap="RdBu_r", vmin=-1, vmax=1).format("{:.2f}"),
        use_container_width=True,
    )

    st.subheader("Pipeline Maps")
    image_cols = st.columns(3)
    images = [
        ("Suitability map", "suitability_map.png"),
        ("Flood mask", "flood_mask.png"),
        ("Terrain features", "terrain_features.png"),
    ]
    for col, (caption, image_name) in zip(image_cols, images):
        with col:
            image_bytes = map_images.get(image_name)
            if image_bytes:
                st.image(image_bytes, caption=caption, use_container_width=True)
            else:
                st.warning(f"Missing {html.escape(image_name)}")


if __name__ == "__main__":
    main()

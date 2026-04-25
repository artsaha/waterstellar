from __future__ import annotations

import hashlib
import json
import math
import os
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CDSE_BASE_URL = "https://sh.dataspace.copernicus.eu"
CDSE_TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
_SENTINEL1_CDSE_COLLECTION = None
SENTINEL1_EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: [{ bands: ["VV", "VH", "dataMask"] }],
    output: { bands: 3, sampleType: "FLOAT32" }
  };
}
function evaluatePixel(s) {
  return [s.VV, s.VH, s.dataMask];
}
"""


def normalize01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype="float64")
    lo = np.nanmin(values)
    hi = np.nanmax(values)
    if math.isclose(float(hi), float(lo)):
        return np.zeros_like(values, dtype="float64")
    return (values - lo) / (hi - lo)


def bounds_slug(name: str, bounds: list[float]) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in name).strip("_")
    digest = hashlib.sha1(",".join(f"{value:.6f}" for value in bounds).encode("utf-8")).hexdigest()[:8]
    return f"aoi_{cleaned or 'custom'}_{digest}"


def parse_bounds(raw: str) -> list[float]:
    parts = [part.strip() for part in raw.replace(";", ",").split(",") if part.strip()]
    if len(parts) != 4:
        raise ValueError("Bounding box must contain four comma-separated values: min_lon,min_lat,max_lon,max_lat.")
    bounds = [float(part) for part in parts]
    min_lon, min_lat, max_lon, max_lat = bounds
    if not (-180 <= min_lon < max_lon <= 180):
        raise ValueError("Longitude bounds must satisfy -180 <= min_lon < max_lon <= 180.")
    if not (-90 <= min_lat < max_lat <= 90):
        raise ValueError("Latitude bounds must satisfy -90 <= min_lat < max_lat <= 90.")
    return bounds


def estimate_area_km2(bounds: list[float]) -> float:
    lon_min, lat_min, lon_max, lat_max = bounds
    radius_km = 6371.0088
    deg = math.pi / 180
    lon_span = (lon_max - lon_min) * deg
    sin_lat_span = math.sin(lat_max * deg) - math.sin(lat_min * deg)
    return abs(radius_km * radius_km * lon_span * sin_lat_span)


def estimate_width_height_km(bounds: list[float]) -> tuple[float, float]:
    lon_min, lat_min, lon_max, lat_max = bounds
    radius_km = 6371.0088
    deg = math.pi / 180
    mid_lat = ((lat_min + lat_max) / 2) * deg
    width = radius_km * ((lon_max - lon_min) * deg) * math.cos(mid_lat)
    height = radius_km * ((lat_max - lat_min) * deg)
    return abs(width), abs(height)


def estimate_flood_fill_metrics(
    flood_mask: np.ndarray,
    dem: np.ndarray,
    bounds: list[float],
    rainfall_intensity_mm_h: float = 25.0,
    runoff_coefficient: float = 0.55,
) -> dict:
    flood_mask = np.asarray(flood_mask, dtype=bool)
    dem = _fill_invalid_dem(dem)
    width_km, height_km = estimate_width_height_km(bounds)
    rows, cols = flood_mask.shape
    cell_area_m2 = (width_km * 1000 / cols) * (height_km * 1000 / rows)
    flood_cell_count = int(flood_mask.sum())
    flood_area_km2 = flood_cell_count * cell_area_m2 / 1_000_000
    effective_rainfall_m_h = max(rainfall_intensity_mm_h, 0.0) / 1000 * max(runoff_coefficient, 0.0)

    if flood_cell_count == 0 or effective_rainfall_m_h <= 0:
        return {
            "fill_hours": None,
            "flood_area_km2": flood_area_km2,
            "storage_volume_m3": 0.0,
            "mean_storage_depth_m": 0.0,
            "spill_elevation_m": None,
            "rainfall_intensity_mm_h": rainfall_intensity_mm_h,
            "runoff_coefficient": runoff_coefficient,
            "method": "Direct rainfall equivalent over DEM-derived flood mask; no hydrodynamic routing.",
        }

    flood_dem = dem[flood_mask]
    spill_elevation_m = float(np.nanpercentile(flood_dem, 90))
    storage_depth_m = np.clip(spill_elevation_m - flood_dem, 0.05, 5.0)
    storage_volume_m3 = float(np.sum(storage_depth_m) * cell_area_m2)
    mean_storage_depth_m = float(storage_volume_m3 / (flood_cell_count * cell_area_m2))
    fill_hours = float(storage_volume_m3 / (effective_rainfall_m_h * flood_cell_count * cell_area_m2))

    return {
        "fill_hours": fill_hours,
        "flood_area_km2": flood_area_km2,
        "storage_volume_m3": storage_volume_m3,
        "mean_storage_depth_m": mean_storage_depth_m,
        "spill_elevation_m": spill_elevation_m,
        "rainfall_intensity_mm_h": rainfall_intensity_mm_h,
        "runoff_coefficient": runoff_coefficient,
        "method": "Direct rainfall equivalent over DEM-derived flood mask; no hydrodynamic routing.",
    }


def _component_from_seed(mask: np.ndarray, seed: tuple[int, int]) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    connected = np.zeros(mask.shape, dtype=bool)
    row, col = seed
    if not mask[row, col]:
        return connected

    stack = [(row, col)]
    connected[row, col] = True
    height, width = mask.shape
    while stack:
        row, col = stack.pop()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                nr, nc = row + dy, col + dx
                if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not connected[nr, nc]:
                    connected[nr, nc] = True
                    stack.append((nr, nc))
    return connected


def _storage_volume_for_mask(
    flood_mask: np.ndarray,
    dem: np.ndarray,
    bounds: list[float],
) -> tuple[float, float, float]:
    flood_mask = np.asarray(flood_mask, dtype=bool)
    if not flood_mask.any():
        return 0.0, 0.0, 0.0

    width_km, height_km = estimate_width_height_km(bounds)
    rows, cols = flood_mask.shape
    cell_area_m2 = (width_km * 1000 / cols) * (height_km * 1000 / rows)
    flood_dem = dem[flood_mask]
    spill_elevation_m = float(np.nanpercentile(flood_dem, 90))
    storage_depth_m = np.clip(spill_elevation_m - flood_dem, 0.05, 5.0)
    storage_volume_m3 = float(np.sum(storage_depth_m) * cell_area_m2)
    mean_storage_depth_m = float(storage_volume_m3 / (int(flood_mask.sum()) * cell_area_m2))
    return storage_volume_m3, mean_storage_depth_m, spill_elevation_m


def _triangular_hydrograph(
    base_discharge_m3_s: float,
    peak_discharge_m3_s: float,
    duration_h: int,
) -> list[float]:
    duration_h = max(int(duration_h), 1)
    base = max(float(base_discharge_m3_s), 0.0)
    peak = max(float(peak_discharge_m3_s), base)
    midpoint = max(duration_h / 2, 0.5)
    values = []
    for hour in range(duration_h + 1):
        if hour <= midpoint:
            value = base + (peak - base) * (hour / midpoint)
        else:
            value = base + (peak - base) * max((duration_h - hour) / (duration_h - midpoint), 0.0)
        values.append(float(value))
    return values


def _first_fill_hour(
    storage_volume_m3: float,
    hourly_inflow_m3_s: list[float],
) -> float | None:
    if storage_volume_m3 <= 0:
        return 0.0
    if not hourly_inflow_m3_s:
        return None

    cumulative_m3 = 0.0
    previous = max(hourly_inflow_m3_s[0], 0.0)
    for hour, current_raw in enumerate(hourly_inflow_m3_s[1:], start=1):
        current = max(current_raw, 0.0)
        step_volume = (previous + current) * 0.5 * 3600
        if cumulative_m3 + step_volume >= storage_volume_m3 and step_volume > 0:
            fraction = (storage_volume_m3 - cumulative_m3) / step_volume
            return (hour - 1) + max(0.0, min(1.0, fraction))
        cumulative_m3 += step_volume
        previous = current
    return None


def _hydrograph_fill_summary(
    storage_volume_m3: float,
    hourly_inflow_m3_s: list[float],
) -> tuple[float | None, float, float, float]:
    if storage_volume_m3 <= 0:
        return 0.0, 0.0, 1.0, 0.0
    if not hourly_inflow_m3_s:
        return None, 0.0, 0.0, storage_volume_m3

    cumulative_m3 = 0.0
    fill_hours = None
    previous = max(hourly_inflow_m3_s[0], 0.0)
    for hour, current_raw in enumerate(hourly_inflow_m3_s[1:], start=1):
        current = max(current_raw, 0.0)
        step_volume = (previous + current) * 0.5 * 3600
        if fill_hours is None and cumulative_m3 + step_volume >= storage_volume_m3 and step_volume > 0:
            fraction = (storage_volume_m3 - cumulative_m3) / step_volume
            fill_hours = (hour - 1) + max(0.0, min(1.0, fraction))
        cumulative_m3 += step_volume
        previous = current

    fill_fraction = min(cumulative_m3 / storage_volume_m3, 1.0)
    volume_deficit_m3 = max(storage_volume_m3 - cumulative_m3, 0.0)
    return fill_hours, cumulative_m3, fill_fraction, volume_deficit_m3


def estimate_routed_flood_fill_metrics(
    flood_mask: np.ndarray,
    dem: np.ndarray,
    flow_accum: np.ndarray,
    bounds: list[float],
    rainfall_intensity_mm_h: float = 25.0,
    runoff_coefficient: float = 0.55,
    base_discharge_m3_s: float = 20.0,
    peak_discharge_m3_s: float = 120.0,
    hydrograph_duration_h: int = 72,
    capture_fraction: float = 0.08,
) -> dict:
    flood_mask = np.asarray(flood_mask, dtype=bool)
    dem = _fill_invalid_dem(dem)
    flow_accum = np.asarray(flow_accum, dtype="float64")
    width_km, height_km = estimate_width_height_km(bounds)
    rows, cols = flood_mask.shape
    cell_area_m2 = (width_km * 1000 / cols) * (height_km * 1000 / rows)
    flood_cell_count = int(flood_mask.sum())

    if flood_cell_count == 0:
        return {
            "fill_hours": None,
            "connected_fill_hours": None,
            "connected_fraction": 0.0,
            "routed_storage_volume_m3": 0.0,
            "captured_peak_discharge_m3_s": 0.0,
            "base_discharge_m3_s": base_discharge_m3_s,
            "peak_discharge_m3_s": peak_discharge_m3_s,
            "hydrograph_duration_h": hydrograph_duration_h,
            "capture_fraction": capture_fraction,
            "method": "DEM D8 connectivity plus triangular river-discharge hydrograph; GloFAS-ready approximation.",
        }

    seed_flat = int(np.argmax(np.where(flood_mask, flow_accum, -np.inf)))
    seed = divmod(seed_flat, cols)
    connected_mask = _component_from_seed(flood_mask, seed)
    connected_fraction = float(connected_mask.sum() / flood_cell_count)
    routed_storage_volume_m3, mean_storage_depth_m, spill_elevation_m = _storage_volume_for_mask(connected_mask, dem, bounds)

    effective_rainfall_m_h = max(rainfall_intensity_mm_h, 0.0) / 1000 * max(runoff_coefficient, 0.0)
    local_runoff_m3_s = effective_rainfall_m_h * int(connected_mask.sum()) * cell_area_m2 / 3600
    flow_accum_ratio = float(
        np.nanmax(flow_accum[connected_mask]) / max(np.nanpercentile(flow_accum, 99), 1.0)
    )
    routing_factor = max(0.05, min(1.0, 0.5 * connected_fraction + 0.5 * flow_accum_ratio))
    effective_capture_fraction = max(0.0, min(1.0, capture_fraction)) * routing_factor
    hydrograph = _triangular_hydrograph(base_discharge_m3_s, peak_discharge_m3_s, hydrograph_duration_h)
    captured_inflow = [q * effective_capture_fraction + local_runoff_m3_s for q in hydrograph]
    fill_hours, routed_volume_m3, fill_fraction, volume_deficit_m3 = _hydrograph_fill_summary(
        routed_storage_volume_m3,
        captured_inflow,
    )

    return {
        "fill_hours": fill_hours,
        "connected_fill_hours": fill_hours,
        "fill_fraction": fill_fraction,
        "routed_volume_m3": routed_volume_m3,
        "volume_deficit_m3": volume_deficit_m3,
        "connected_fraction": connected_fraction,
        "routed_storage_volume_m3": routed_storage_volume_m3,
        "mean_storage_depth_m": mean_storage_depth_m,
        "spill_elevation_m": spill_elevation_m,
        "local_runoff_m3_s": float(local_runoff_m3_s),
        "captured_peak_discharge_m3_s": float(max(captured_inflow) if captured_inflow else 0.0),
        "effective_capture_fraction": float(effective_capture_fraction),
        "base_discharge_m3_s": base_discharge_m3_s,
        "peak_discharge_m3_s": peak_discharge_m3_s,
        "hydrograph_duration_h": hydrograph_duration_h,
        "capture_fraction": capture_fraction,
        "routing_factor": routing_factor,
        "inlet_row": int(seed[0]),
        "inlet_col": int(seed[1]),
        "method": "DEM D8 connectivity plus triangular river-discharge hydrograph; GloFAS-ready approximation.",
    }


def _seed_from_bounds(bounds: list[float]) -> int:
    digest = hashlib.sha1(",".join(f"{value:.6f}" for value in bounds).encode("utf-8")).hexdigest()[:8]
    return int(digest, 16)


def _db_to_linear(db: np.ndarray) -> np.ndarray:
    return np.power(10.0, db / 10.0)


def _to_db(linear: np.ndarray) -> np.ndarray:
    return 10 * np.log10(np.clip(linear, 1e-6, None))


def _load_local_env(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
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


def _sentinelhub_config():
    try:
        from sentinelhub import SHConfig
    except ImportError as exc:
        raise ImportError("sentinelhub is required to download Copernicus DEM GLO-30") from exc

    _load_local_env()
    client_id = (os.getenv("SH_CLIENT_ID") or "").strip()
    client_secret = (os.getenv("SH_CLIENT_SECRET") or "").strip()
    if not client_id or not client_secret:
        raise RuntimeError("Set both SH_CLIENT_ID and SH_CLIENT_SECRET in Streamlit secrets, environment variables, or .env to download Copernicus DEM GLO-30.")

    config = SHConfig()
    config.sh_client_id = client_id
    config.sh_client_secret = client_secret
    config.sh_base_url = CDSE_BASE_URL
    config.sh_token_url = CDSE_TOKEN_URL
    return config


def _cdse_dem_collection():
    from sentinelhub import DataCollection

    existing = getattr(DataCollection, "DEM_CDSE", None)
    if existing is not None:
        return existing
    return DataCollection.DEM.define_from("DEM_CDSE", service_url=CDSE_BASE_URL)


def _cdse_sentinel1_collection():
    global _SENTINEL1_CDSE_COLLECTION
    if _SENTINEL1_CDSE_COLLECTION is not None:
        return _SENTINEL1_CDSE_COLLECTION

    from sentinelhub import DataCollection

    _SENTINEL1_CDSE_COLLECTION = DataCollection.SENTINEL1_IW.define_from(
        "SENTINEL1_IW_CDSE_WEB_PAPER",
        service_url=CDSE_BASE_URL,
    )
    return _SENTINEL1_CDSE_COLLECTION


def _download_copernicus_dem_glo30(
    bounds: list[float],
    height: int,
    width: int,
    resolution_m: float,
) -> np.ndarray:
    from sentinelhub import BBox, CRS, MimeType, SentinelHubRequest, bbox_to_dimensions

    evalscript_dem = """
//VERSION=3
function setup() {
  return {
    input: [{ bands: ["DEM"] }],
    output: { bands: 1, sampleType: "FLOAT32" }
  };
}
function evaluatePixel(s) {
  return [s.DEM];
}
"""

    config = _sentinelhub_config()
    bbox = BBox(bbox=bounds, crs=CRS.WGS84)
    size = bbox_to_dimensions(bbox, resolution=resolution_m)
    pixel_count = size[0] * size[1]
    if pixel_count > 2_500_000:
        raise ValueError("AOI is too large for one DEM request. Increase resolution or split the AOI into tiles.")

    request = SentinelHubRequest(
        evalscript=evalscript_dem,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=_cdse_dem_collection(),
                other_args={"dataFilter": {"demInstance": "COPERNICUS_30"}},
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=size,
        config=config,
    )
    data = request.get_data()[0]
    if data.ndim == 3:
        data = data[:, :, 0]
    if data.shape[:2] != (size[1], size[0]):
        raise RuntimeError(f"Unexpected Copernicus DEM response shape: {data.shape}")
    dem = _fill_invalid_dem(data)
    if dem.shape != (height, width):
        dem = _resize_array(dem, height, width)
    return dem


def _fetch_sentinel1_patch(
    bounds: list[float],
    height: int,
    width: int,
    time_interval: tuple[str, str],
) -> np.ndarray:
    from sentinelhub import BBox, CRS, MimeType, SentinelHubRequest

    config = _sentinelhub_config()
    bbox = BBox(bbox=bounds, crs=CRS.WGS84)
    request = SentinelHubRequest(
        evalscript=SENTINEL1_EVALSCRIPT,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=_cdse_sentinel1_collection(),
                time_interval=time_interval,
                other_args={
                    "processing": {
                        "backCoeff": "SIGMA0_ELLIPSOID",
                        "orthorectify": True,
                    }
                },
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=(width, height),
        config=config,
    )
    data = request.get_data()[0]
    if data.shape[:2] != (height, width) or data.shape[2] < 2:
        raise RuntimeError(f"Unexpected Sentinel-1 response shape: {data.shape}")
    mask = data[:, :, 2] > 0 if data.shape[2] >= 3 else np.isfinite(data[:, :, 0])
    vv = np.where(mask, data[:, :, 0], np.nan)
    vh = np.where(mask, data[:, :, 1], np.nan)
    if not np.isfinite(vv).any() or not np.isfinite(vh).any():
        raise RuntimeError(f"No finite Sentinel-1 VV/VH pixels returned for interval {time_interval}")
    vv = np.where(np.isfinite(vv), vv, np.nanmedian(vv))
    vh = np.where(np.isfinite(vh), vh, np.nanmedian(vh))
    return np.dstack([vv, vh]).astype("float32")


def _make_sentinel1_sar(
    height: int,
    width: int,
    bounds: list[float],
    dry_interval: tuple[str, str],
    wet_interval: tuple[str, str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    dry = _fetch_sentinel1_patch(bounds, height, width, dry_interval)
    wet = _fetch_sentinel1_patch(bounds, height, width, wet_interval)
    vv_dry_db = _to_db(dry[:, :, 0])
    vv_wet_db = _to_db(wet[:, :, 0])
    vh_wet_db = _to_db(wet[:, :, 1])
    backscatter_drop = vv_dry_db - vv_wet_db
    wet_water_like = vv_wet_db < -15.0
    cross_pol_low = vh_wet_db < -20.0
    flood_mask = (backscatter_drop > 3.0) & (wet_water_like | cross_pol_low)
    source = f"live Sentinel-1 IW VV/VH via Sentinel Hub: dry {dry_interval[0]} to {dry_interval[1]}, wet {wet_interval[0]} to {wet_interval[1]}"
    return dry, wet, flood_mask, source


def _generate_dem(height: int, width: int, bounds: list[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a synthetic fallback DEM without the old repeating sine valley."""
    rng = np.random.default_rng(_seed_from_bounds(bounds))
    y, x = np.mgrid[0:height, 0:width]
    xn = x / width
    yn = y / height

    min_lon, min_lat, max_lon, max_lat = bounds
    lat_mid = 0.5 * (min_lat + max_lat)
    aspect = max((max_lon - min_lon) / max(max_lat - min_lat, 1e-9), 0.25)

    slope_angle = rng.uniform(0, 2 * np.pi)
    slope_strength = rng.uniform(55, 120)
    slope_axis = np.cos(slope_angle) * (xn - 0.5) + np.sin(slope_angle) * (yn - 0.5)
    dem = 850 + 0.35 * lat_mid + slope_strength * slope_axis
    dem += rng.uniform(10, 35) * np.sin(rng.uniform(2.2, 5.5) * np.pi * xn + rng.uniform(0, 2 * np.pi))
    dem += rng.uniform(8, 28) * np.cos(rng.uniform(2.0, 5.0) * np.pi * yn + rng.uniform(0, 2 * np.pi))

    for _ in range(rng.integers(5, 9)):
        bx = rng.uniform(0.10, 0.90)
        by = rng.uniform(0.10, 0.90)
        rx = rng.uniform(0.012, 0.055) * np.clip(aspect, 0.7, 1.8)
        ry = rng.uniform(0.010, 0.050)
        dem -= rng.uniform(18, 70) * np.exp(-(((xn - bx) ** 2) / rx + ((yn - by) ** 2) / ry))

    roughness = rng.uniform(7, 16)
    dem += _smooth_array(rng.normal(0, roughness, (height, width)), radius=3)
    dem += _smooth_array(rng.normal(0, roughness * 0.6, (height, width)), radius=10)
    return dem, xn, yn


def _fill_invalid_dem(dem: np.ndarray) -> np.ndarray:
    dem = np.asarray(dem, dtype="float64")
    invalid = ~np.isfinite(dem)
    if invalid.all():
        raise ValueError("DEM contains no finite elevation values for this AOI")
    if invalid.any():
        dem = dem.copy()
        dem[invalid] = np.nanmedian(dem)
    return dem


def _load_local_dem(
    height: int,
    width: int,
    bounds: list[float],
    dem_path: str | Path | None = None,
    dem_bytes: bytes | None = None,
) -> tuple[np.ndarray, str]:
    path = Path(dem_path).expanduser() if dem_path is not None else None
    if dem_bytes is None and (path is None or not path.exists()):
        raise FileNotFoundError(f"No local DEM found at {path}")

    try:
        import rasterio
        from rasterio.io import MemoryFile
        from rasterio.transform import from_bounds
        from rasterio.warp import Resampling, reproject
    except ImportError as exc:
        raise ImportError("rasterio is required to read a local DEM GeoTIFF") from exc

    min_lon, min_lat, max_lon, max_lat = bounds
    destination = np.full((height, width), np.nan, dtype="float32")
    dst_transform = from_bounds(min_lon, min_lat, max_lon, max_lat, width, height)

    if dem_bytes is not None:
        source_label = "uploaded Copernicus DEM GLO-30 GeoTIFF"
        with MemoryFile(dem_bytes) as memfile:
            with memfile.open() as src:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=destination,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    src_nodata=src.nodata,
                    dst_transform=dst_transform,
                    dst_crs="EPSG:4326",
                    dst_nodata=np.nan,
                    resampling=Resampling.bilinear,
                )
    else:
        source_label = f"local Copernicus DEM GLO-30 GeoTIFF: {path}"
        with rasterio.open(path) as src:
            reproject(
                source=rasterio.band(src, 1),
                destination=destination,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src.nodata,
                dst_transform=dst_transform,
                dst_crs="EPSG:4326",
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )
    return _fill_invalid_dem(destination), source_label


def _load_dem_for_aoi(
    height: int,
    width: int,
    bounds: list[float],
    dem_path: str | Path | None = None,
    dem_bytes: bytes | None = None,
    require_real_dem: bool = False,
    download_dem: bool = False,
    dem_resolution_m: float = 30,
) -> tuple[np.ndarray, str, bool]:
    errors = []
    if dem_bytes is not None:
        try:
            dem, source = _load_local_dem(height, width, bounds, dem_bytes=dem_bytes)
            return dem, source, True
        except Exception as exc:
            errors.append(str(exc))

    candidate_path = dem_path or os.getenv("COPERNICUS_DEM_PATH")
    if candidate_path:
        try:
            dem, source = _load_local_dem(height, width, bounds, candidate_path)
            return dem, source, True
        except Exception as exc:
            errors.append(str(exc))

    if download_dem:
        try:
            dem = _download_copernicus_dem_glo30(bounds, height, width, dem_resolution_m)
            return dem, "downloaded Copernicus DEM GLO-30 via Sentinel Hub", True
        except Exception as exc:
            errors.append(f"DEM download failed: {exc}")

    if require_real_dem:
        detail = " | ".join(errors) if errors else "No COPERNICUS_DEM_PATH or DEM file was provided"
        raise RuntimeError(f"Real DEM required but unavailable. {detail}")

    dem, _, _ = _generate_dem(height, width, bounds)
    return dem, "synthetic fallback DEM", False


def _smooth_array(values: np.ndarray, radius: int) -> np.ndarray:
    padded = np.pad(values, radius, mode="edge")
    smoothed = np.zeros_like(values, dtype="float64")
    size = 2 * radius + 1
    for dy in range(size):
        for dx in range(size):
            smoothed += padded[dy : dy + values.shape[0], dx : dx + values.shape[1]]
    return smoothed / (size * size)


def _resize_array(values: np.ndarray, height: int, width: int) -> np.ndarray:
    values = np.asarray(values, dtype="float64")
    src_y = np.linspace(0.0, 1.0, values.shape[0])
    src_x = np.linspace(0.0, 1.0, values.shape[1])
    dst_y = np.linspace(0.0, 1.0, height)
    dst_x = np.linspace(0.0, 1.0, width)
    row_resampled = np.vstack([np.interp(dst_x, src_x, row) for row in values])
    return np.vstack([np.interp(dst_y, src_y, row_resampled[:, col]) for col in range(width)]).T


def _make_synthetic_sar(
    height: int,
    width: int,
    bounds: list[float],
    dem_path: str | Path | None = None,
    dem_bytes: bytes | None = None,
    require_real_dem: bool = False,
    download_dem: bool = False,
    dem_resolution_m: float = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, bool]:
    rng = np.random.default_rng(_seed_from_bounds(bounds))
    dem, dem_source, is_real_dem = _load_dem_for_aoi(
        height,
        width,
        bounds,
        dem_path=dem_path,
        dem_bytes=dem_bytes,
        require_real_dem=require_real_dem,
        download_dem=download_dem,
        dem_resolution_m=dem_resolution_m,
    )

    dry_vv_db = -8.5 + 1.8 * rng.standard_normal((height, width))
    dry_vh_db = -14.0 + 2.0 * rng.standard_normal((height, width))

    smoothed_dem = _smooth_array(dem, radius=15)
    relative_elevation = dem - smoothed_dem
    depression_cutoff = np.percentile(relative_elevation, rng.uniform(8, 16))
    lowland_cutoff = np.percentile(dem, rng.uniform(8, 14))
    water_mask = (relative_elevation < depression_cutoff) | ((dem < lowland_cutoff) & (relative_elevation < 0))

    wet_vv_db = dry_vv_db + rng.normal(0.0, 0.6, (height, width))
    wet_vh_db = dry_vh_db + rng.normal(0.0, 0.7, (height, width))
    wet_vv_db[water_mask] -= rng.uniform(4.0, 9.0, water_mask.sum())
    wet_vh_db[water_mask] -= rng.uniform(2.0, 5.0, water_mask.sum())

    dry = np.dstack([_db_to_linear(dry_vv_db), _db_to_linear(dry_vh_db)]).astype("float32")
    wet = np.dstack([_db_to_linear(wet_vv_db), _db_to_linear(wet_vh_db)]).astype("float32")
    return dry, wet, water_mask, dem, dem_source, is_real_dem


def _compute_flow_accumulation(dem: np.ndarray) -> np.ndarray:
    dem = _fill_invalid_dem(dem)
    height, width = dem.shape
    flow_accum = np.ones((height, width), dtype="float64")
    order = np.argsort(dem.ravel())[::-1]
    neighbor_offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    for flat_index in order:
        row, col = divmod(int(flat_index), width)
        current = dem[row, col]
        best_row, best_col = row, col
        best_drop = 0.0
        for dy, dx in neighbor_offsets:
            nr, nc = row + dy, col + dx
            if 0 <= nr < height and 0 <= nc < width:
                drop = current - dem[nr, nc]
                if drop > best_drop:
                    best_drop = drop
                    best_row, best_col = nr, nc
        if best_drop > 0:
            flow_accum[best_row, best_col] += flow_accum[row, col]
    return flow_accum


def _terrain_derivatives_from_dem(
    dem: np.ndarray,
    resolution_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dem = _fill_invalid_dem(dem)
    grad_y, grad_x = np.gradient(dem, resolution_m)
    slope = np.degrees(np.arctan(np.hypot(grad_x, grad_y)))

    local_mean = _smooth_array(dem, radius=6)
    tpi = dem - local_mean

    flow_accum = _compute_flow_accumulation(dem)
    slope_rad = np.radians(np.maximum(slope, 0.1))
    twi = np.log(((flow_accum + 1.0) * resolution_m) / (np.tan(slope_rad) + 1e-6))

    grad_yy, _ = np.gradient(grad_y, resolution_m)
    _, grad_xx = np.gradient(grad_x, resolution_m)
    curvature = -(grad_xx + grad_yy)
    return dem, slope, tpi, twi, flow_accum, curvature


def pixel_to_lonlat(row: int, col: int, bounds: list[float], shape: tuple[int, int]) -> tuple[float, float]:
    height, width = shape
    min_lon, min_lat, max_lon, max_lat = bounds
    lon = min_lon + (col + 0.5) * (max_lon - min_lon) / width
    lat = max_lat - (row + 0.5) * (max_lat - min_lat) / height
    return lon, lat


def compute_candidates(
    bounds: list[float],
    name: str = "custom",
    height: int = 300,
    width: int = 300,
    top_n: int = 250,
    resolution_m: float = 40,
    dem_path: str | Path | None = None,
    dem_bytes: bytes | None = None,
    require_real_dem: bool = False,
    download_dem: bool = False,
    dem_resolution_m: float = 30,
    rainfall_intensity_mm_h: float = 25.0,
    runoff_coefficient: float = 0.55,
    routed_base_discharge_m3_s: float = 20.0,
    routed_peak_discharge_m3_s: float = 120.0,
    routed_hydrograph_duration_h: int = 72,
    routed_capture_fraction: float = 0.08,
    use_live_sentinel1: bool = False,
    sentinel1_dry_start: str = "2023-01-01",
    sentinel1_dry_end: str = "2023-01-31",
    sentinel1_wet_start: str = "2023-04-01",
    sentinel1_wet_end: str = "2023-04-30",
) -> dict:
    run_id = bounds_slug(name, bounds)

    if use_live_sentinel1:
        dem, dem_source, is_real_dem = _load_dem_for_aoi(
            height,
            width,
            bounds,
            dem_path=dem_path,
            dem_bytes=dem_bytes,
            require_real_dem=require_real_dem,
            download_dem=download_dem,
            dem_resolution_m=dem_resolution_m,
        )
        dry, wet, flood_mask, flood_data_source = _make_sentinel1_sar(
            height,
            width,
            bounds,
            dry_interval=(sentinel1_dry_start, sentinel1_dry_end),
            wet_interval=(sentinel1_wet_start, sentinel1_wet_end),
        )
        mode = "sentinel-1-live-sar-dem-terrain"
    else:
        dry, wet, flood_mask, dem, dem_source, is_real_dem = _make_synthetic_sar(
            height,
            width,
            bounds,
            dem_path=dem_path,
            dem_bytes=dem_bytes,
            require_real_dem=require_real_dem,
            download_dem=download_dem,
            dem_resolution_m=dem_resolution_m,
        )
        flood_data_source = "DEM-driven synthetic SAR"
        mode = "dem-driven-synthetic-sar"

    vv_dry_db = _to_db(dry[:, :, 0])
    vv_wet_db = _to_db(wet[:, :, 0])
    backscatter_diff = vv_dry_db - vv_wet_db

    dem, slope, tpi, twi, flow_accum, curvature = _terrain_derivatives_from_dem(dem, resolution_m)
    flood_fill_metrics = estimate_flood_fill_metrics(
        flood_mask,
        dem,
        bounds,
        rainfall_intensity_mm_h=rainfall_intensity_mm_h,
        runoff_coefficient=runoff_coefficient,
    )
    routed_flood_fill_metrics = estimate_routed_flood_fill_metrics(
        flood_mask,
        dem,
        flow_accum,
        bounds,
        rainfall_intensity_mm_h=rainfall_intensity_mm_h,
        runoff_coefficient=runoff_coefficient,
        base_discharge_m3_s=routed_base_discharge_m3_s,
        peak_discharge_m3_s=routed_peak_discharge_m3_s,
        hydrograph_duration_h=routed_hydrograph_duration_h,
        capture_fraction=routed_capture_fraction,
    )
    seasonality_ratio = 4.6
    suitability_map = (
        0.30 * normalize01(backscatter_diff)
        + 0.20 * (1 - normalize01(slope))
        + 0.20 * normalize01(twi)
        + 0.15 * normalize01(flow_accum)
        + 0.10 * normalize01(-tpi)
        + 0.05 * min(seasonality_ratio / 6.0, 1.0)
    )
    suitability_map = np.clip(suitability_map, 0, 1)
    flood_frequency = flood_mask.astype("float64")

    rows, cols = np.indices(suitability_map.shape)
    flat = pd.DataFrame(
        {
            "row": rows.ravel(),
            "col": cols.ravel(),
            "score": suitability_map.ravel(),
            "flood_frequency": flood_frequency.ravel(),
            "slope": slope.ravel(),
            "twi": twi.ravel(),
            "tpi": tpi.ravel(),
            "flow_accum": flow_accum.ravel(),
        }
    ).sort_values("score", ascending=False)
    flooded_flat = flat[flat["flood_frequency"] > 0]
    if len(flooded_flat) >= top_n:
        flat = flooded_flat
    top = flat.head(top_n).copy()
    coords = [pixel_to_lonlat(int(r), int(c), bounds, suitability_map.shape) for r, c in zip(top["row"], top["col"])]
    top["lon"] = [coord[0] for coord in coords]
    top["lat"] = [coord[1] for coord in coords]
    top = top[["lon", "lat", "row", "col", "score", "flood_frequency", "slope", "twi", "tpi", "flow_accum"]]

    features = []
    for record in top.to_dict(orient="records"):
        lon = float(record.pop("lon"))
        lat = float(record.pop("lat"))
        properties = {key: float(value) if isinstance(value, (np.floating, float)) else int(value) for key, value in record.items()}
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": properties,
            }
        )
    geojson = {"type": "FeatureCollection", "features": features}

    map_images = _render_maps(flood_mask, suitability_map, dem, slope, tpi, twi, flow_accum, curvature, dem_source)
    metadata = {
        "name": name,
        "run_id": run_id,
        "bounds": bounds,
        "mode": mode,
        "dem_source": dem_source,
        "is_real_dem": is_real_dem,
        "flood_data_source": flood_data_source,
        "use_live_sentinel1": use_live_sentinel1,
        "sentinel1_dry_interval": [sentinel1_dry_start, sentinel1_dry_end],
        "sentinel1_wet_interval": [sentinel1_wet_start, sentinel1_wet_end],
        "download_dem": download_dem,
        "dem_resolution_m": dem_resolution_m,
        "height": height,
        "width": width,
        "top_n": top_n,
        "flood_fill": flood_fill_metrics,
        "routed_flood_fill": routed_flood_fill_metrics,
        "area_km2": estimate_area_km2(bounds),
    }
    return {
        "run_id": run_id,
        "candidates": top,
        "geojson": geojson,
        "metadata": metadata,
        "map_images": map_images,
        "bounds": bounds,
        "dem_source": dem_source,
        "is_real_dem": is_real_dem,
        "flood_data_source": flood_data_source,
        "flood_fill": flood_fill_metrics,
        "routed_flood_fill": routed_flood_fill_metrics,
    }


def _png_bytes_from_fig(fig: plt.Figure) -> bytes:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=150)
    plt.close(fig)
    return buffer.getvalue()


def _render_maps(
    flood_mask: np.ndarray,
    suitability_map: np.ndarray,
    dem: np.ndarray,
    slope: np.ndarray,
    tpi: np.ndarray,
    twi: np.ndarray,
    flow_accum: np.ndarray,
    curvature: np.ndarray,
    dem_source: str,
) -> dict[str, bytes]:
    flood_overlay = np.zeros((*flood_mask.shape, 4), dtype="float32")
    flood_overlay[flood_mask] = [0.10, 0.38, 0.95, 0.70]
    overlay_buffer = BytesIO()
    plt.imsave(overlay_buffer, flood_overlay, format="png")
    images = {"flood_zones_overlay.png": overlay_buffer.getvalue()}

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(flood_mask, cmap="Blues")
    ax.set_title("Flood mask")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    images["flood_mask.png"] = _png_bytes_from_fig(fig)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(suitability_map, cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_title("Suitability score")
    ax.axis("off")
    plt.colorbar(im, ax=ax, label="Score 0-1", fraction=0.046, pad=0.04)
    fig.tight_layout()
    images["suitability_map.png"] = _png_bytes_from_fig(fig)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    layers = [dem, slope, tpi, twi, flow_accum, curvature]
    titles = ["DEM", "Slope", "TPI", "TWI", "Flow accumulation", "Curvature"]
    cmaps = ["terrain", "magma", "coolwarm", "YlGnBu", "viridis", "PiYG"]
    for ax, layer, title, cmap in zip(axes.ravel(), layers, titles, cmaps):
        im = ax.imshow(layer, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    images["terrain_features.png"] = _png_bytes_from_fig(fig)
    return images

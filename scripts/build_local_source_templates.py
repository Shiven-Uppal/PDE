"""Build 1-km traffic and road-dust source proxies from raw OSM GeoPackage."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyogrio
from scipy.ndimage import gaussian_filter
from shapely.geometry import box

CRS = "EPSG:32643"
X0_M, Y0_M, CELL_M, NX, NY = 682_750.0, 3_145_665.0, 1_000.0, 50, 50
TRAFFIC_WEIGHTS = {"motorway": 1.00, "motorway_link": 0.85, "trunk": 0.90, "trunk_link": 0.75, "primary": 0.75, "primary_link": 0.65, "secondary": 0.60, "secondary_link": 0.50, "tertiary": 0.45, "tertiary_link": 0.38, "unclassified": 0.35, "residential": 0.25, "service": 0.10}
DUST_POLYGON_WHERE = "landuse = 'construction' OR natural IN ('sand', 'bare_rock', 'shingle')"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_grid() -> gpd.GeoDataFrame:
    rows = []
    for j in range(NY):
        for i in range(NX):
            rows.append({"i": i, "j": j, "cell_id": j * NX + i, "geometry": box(X0_M + i * CELL_M, Y0_M + j * CELL_M, X0_M + (i + 1) * CELL_M, Y0_M + (j + 1) * CELL_M)})
    return gpd.GeoDataFrame(rows, crs=CRS)


def line_density(lines: gpd.GeoDataFrame, grid: gpd.GeoDataFrame, weights: dict[str, float]) -> np.ndarray:
    lines = lines.copy()
    lines["weight"] = lines.highway.map(weights).astype(float)
    joined = gpd.sjoin(lines[["weight", "geometry"]], grid[["cell_id", "geometry"]], how="inner", predicate="intersects")
    joined = joined.join(grid.geometry.rename("cell_geometry"), on="index_right")
    length = joined.geometry.intersection(gpd.GeoSeries(joined.cell_geometry, crs=CRS)).length
    return (length * joined.weight).groupby(joined.cell_id).sum().reindex(grid.cell_id, fill_value=0.0).to_numpy()


def area_density(polygons: gpd.GeoDataFrame, grid: gpd.GeoDataFrame) -> np.ndarray:
    if polygons.empty:
        return np.zeros(len(grid))
    joined = gpd.sjoin(polygons[["geometry"]], grid[["cell_id", "geometry"]], how="inner", predicate="intersects")
    joined = joined.join(grid.geometry.rename("cell_geometry"), on="index_right")
    area = joined.geometry.intersection(gpd.GeoSeries(joined.cell_geometry, crs=CRS)).area
    return area.groupby(joined.cell_id).sum().reindex(grid.cell_id, fill_value=0.0).to_numpy()


def normalise(values: np.ndarray) -> np.ndarray:
    """Compress map-feature outliers, then set the spatial mean to one."""
    positive = values[values > 0]
    if len(positive) == 0:
        return np.zeros_like(values)
    compressed = np.log1p(values / np.median(positive))
    return compressed / compressed.mean()


def scale_mean(values: np.ndarray) -> np.ndarray:
    return values / values.mean() if values.mean() > 0 else np.zeros_like(values)


def smooth_template(values: np.ndarray) -> np.ndarray:
    """Apply a fixed 1.5 km isotropic source-footprint regularisation."""
    smoothed = gaussian_filter(values.reshape(NY, NX), sigma=1.5, mode="constant")
    return scale_mean(smoothed.ravel())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geopackage", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    gpkg, output = Path(args.geopackage), Path(args.output_dir)
    grid = make_grid()
    where = "highway IN ({})".format(",".join(repr(x) for x in TRAFFIC_WEIGHTS))
    roads = pyogrio.read_dataframe(gpkg, layer="lines", columns=["highway"], where=where).to_crs(CRS)
    dust_polygons = pyogrio.read_dataframe(gpkg, layer="multipolygons", columns=["landuse", "natural"], where=DUST_POLYGON_WHERE).to_crs(CRS)
    traffic_raw = line_density(roads, grid, TRAFFIC_WEIGHTS)
    dust_road_raw = line_density(roads, grid, {key: 1.0 for key in TRAFFIC_WEIGHTS})
    construction_raw = area_density(dust_polygons[dust_polygons.landuse == "construction"], grid)
    bare_raw = area_density(dust_polygons[dust_polygons.natural.isin(["sand", "bare_rock", "shingle"])], grid)
    components = [normalise(x) for x in (dust_road_raw, construction_raw, bare_raw) if x.sum() > 0]
    dust_raw = np.mean(components, axis=0)
    result = grid.drop(columns="geometry").copy()
    result["x_center_m"] = X0_M + (result.i + 0.5) * CELL_M
    result["y_center_m"] = Y0_M + (result.j + 0.5) * CELL_M
    result["traffic_proxy_raw_m"] = traffic_raw
    result["dust_road_proxy_raw_m"] = dust_road_raw
    result["construction_proxy_raw_m2"] = construction_raw
    result["bare_ground_proxy_raw_m2"] = bare_raw
    result["traffic_template"] = smooth_template(normalise(traffic_raw))
    result["dust_template"] = smooth_template(scale_mean(dust_raw))
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / "local_source_component_templates.csv", index=False)
    pd.DataFrame([{"source_geopackage": gpkg.name, "sha256": sha256(gpkg), "crs": CRS, "domain_x0_m": X0_M, "domain_y0_m": Y0_M, "cell_m": CELL_M, "nx": NX, "ny": NY, "road_features_used": len(roads), "dust_polygons_used": len(dust_polygons), "traffic_template_spatial_mean": result.traffic_template.mean(), "dust_template_spatial_mean": result.dust_template.mean()}]).to_csv(output / "local_source_template_manifest.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for axis, column, title in zip(axes, ["traffic_template", "dust_template"], ["Traffic-road proxy", "Road-dust proxy"]):
        image = axis.imshow(result[column].to_numpy().reshape(NY, NX), origin="lower", cmap="magma")
        axis.set_title(title); axis.set_xlabel("grid x (km)"); axis.set_ylabel("grid y (km)")
        fig.colorbar(image, ax=axis, label="relative source intensity")
    fig.savefig(output / "local_source_component_templates.png", dpi=220)
    print(f"Road features: {len(roads)} | dust polygons: {len(dust_polygons)} | grid cells: {len(result)}")


if __name__ == "__main__":
    main()

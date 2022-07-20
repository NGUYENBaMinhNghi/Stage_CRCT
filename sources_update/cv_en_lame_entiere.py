import json
from argparse import ArgumentParser
from pathlib import Path
import cv2
import os

import geopandas
from pathaia.util.paths import get_files
from pytorch_lightning.utilities.seed import seed_everything
from shapely.affinity import translate
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union
from rasterio import features
from shapely.geometry import MultiPolygon, Polygon, shape
import numpy as np


def get_reduced_coords(coords, angle_th: float, distance_th: float):
    r"""
    Given polygon vertices coordinates, deletes those that are too close or that form
    a too small angle.
    Args:
        coords: array of coordinates.
        angle_th: minimum angle (in degrees) formed by 3 consecutives vertices. If the
            angle is too small, the middle vertex will be deleted.
        distance_th: minimum distance between vertices. If 2 consecutive sides of the
            polygon are too small, the middle vertex will be deleted.
    Returns:
        Array of polygon coordinates with small and flat sides pruned.
    """
    vector_rep = np.diff(coords, axis=0)
    angle_th_rad = np.deg2rad(angle_th)
    points_removed = [0]
    while len(points_removed):
        points_removed = list()
        for i in range(len(vector_rep) - 1):
            if len(coords) - len(points_removed) == 3:
                break
            v01 = vector_rep[i]
            v12 = vector_rep[i + 1]
            d01 = np.linalg.norm(v01)
            d12 = np.linalg.norm(v12)
            if d01 < distance_th and d12 < distance_th:
                points_removed.append(i + 1)
                vector_rep[i + 1] = coords[i + 2] - coords[i]
                continue
            angle = np.arccos(np.dot(v01, v12) / (d01 * d12))
            if angle < angle_th_rad:
                points_removed.append(i + 1)
                vector_rep[i + 1] = coords[i + 2] - coords[i]
        coords = np.delete(coords, points_removed, axis=0)
        vector_rep = np.diff(coords, axis=0)
    return coords.astype(int)


def reduce_polygon(
    polygon: Polygon, angle_th: float = 0, distance_th: float = 0
) -> Polygon:
    r"""
    Given a :class:`shapely.geometry.Polygon`, delete vertices that create small or
    flat sides on the interior and on the exterior.
    Args:
        polygon: input polygon.
        angle_th: minimum angle (in degrees) formed by 3 consecutives vertices. If the
            angle is too small, the middle vertex will be deleted.
        distance_th: minimum distance between vertices. If 2 consecutive sides of the
            polygon are too small, the middle vertex will be deleted.
    Returns:
        Reduced polygon.
    """
    ext_poly_coords = get_reduced_coords(
        np.array(polygon.exterior.coords[:]), angle_th, distance_th
    )
    interior_coords = [
        get_reduced_coords(np.array(interior.coords[:]), angle_th, distance_th)
        for interior in polygon.interiors
    ]
    return Polygon(ext_poly_coords, interior_coords)


def mask_to_polygons_layer(
    mask, angle_th: float = 2, distance_th: float = 3
) -> MultiPolygon:
    """
    Convert mask array into :class:`shapely.geometry.MultiPolygon`.
    Args:
        mask: input mask array.
    Returns:
        :class:`shapely.geometry.MultiPolygon` where polygons are extracted from
        positive areas in the mask.
    """
    all_polygons = []
    for sh, _ in features.shapes(mask.astype(np.int16), mask=(mask > 0)):
        all_polygons.append(
            reduce_polygon(shape(sh), angle_th=angle_th, distance_th=distance_th)
        )

    all_polygons = MultiPolygon(all_polygons)
    if not all_polygons.is_valid:
        all_polygons = all_polygons.buffer(0)
        # Sometimes buffer() converts a simple Multipolygon to just a Polygon,
        # need to keep it a Multi throughout
        if all_polygons.type == "Polygon":
            all_polygons = MultiPolygon([all_polygons])
    return all_polygons


parser = ArgumentParser(prog=(" "))

parser.add_argument(
    "--outfolder", type=Path, help="Output folder for geojsons.", required=True
)

parser.add_argument(
    "--mask_folder",
    type=Path,
    help="Input folder containing masks.",
    required=True,
)

parser.add_argument(
    "--patch-size",
    type=int,
    default=512,
    help="Size of the patches used for training. Default 1024.",
)

parser.add_argument(
    "--area-threshold",
    type=int,
    default=50,
    help="Minimum area of objects to keep. Default 50.",
)

if __name__ == "__main__":
    args = parser.parse_args()

    seed_everything(workers=True)

    mask_paths = get_files(args.mask_folder, extensions=".png", recurse=False).sorted(
        key=lambda x: x.stem
    )
    slide_name = mask_paths[0].stem.split("_")[0] + mask_paths[0].stem.split("_")[1]

    polygons = []
    for mask_path in mask_paths:
        patch_x = int(mask_paths[0].stem.split("_")[2])
        patch_y = int(mask_paths[0].stem.split("_")[3])
        print(mask_path)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        polygon = mask_to_polygons_layer(mask, angle_th=0, distance_th=0)

        polygon = translate(polygon, xoff=patch_x, yoff=patch_y)

        if isinstance(polygon, Polygon) and polygon.area > args.area_threshold:
            polygons.append(polygon)
        elif isinstance(polygon, MultiPolygon):
            for pol in polygon.geoms:
                if pol.area > args.area_threshold:
                    polygons.append(pol)
    polygons = unary_union(polygons)
    if isinstance(polygons, Polygon):
        polygons = MultiPolygon(polygons=[polygons])
    with open(args.outfolder / f"{slide_name}_pred.geojson", "w") as f:
        json.dump(geopandas.GeoSeries(polygons.geoms).__geo_interface__, f)
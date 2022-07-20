import json
from argparse import ArgumentParser
from pathlib import Path
import cv2
import os

OPENSLIDE_PATH = 'C:\\Users\\nguye\\Downloads\\openslide-win64-20171122\\bin\\'
os.environ['PATH'] = OPENSLIDE_PATH + ";" + os.environ['PATH']

from shapely.affinity import translate
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union
from rasterio import features
from shapely.geometry import MultiPolygon, Polygon, shape
import numpy as np
import geopandas

from fastcore.foundation import L, setify
def _get_files(p, fs, extensions=None):
    p = Path(p)
    res = [
        p / f
        for f in fs
        if not f.startswith(".")
        and ((not extensions) or f'.{f.split(".")[-1].lower()}' in extensions)
    ]
    return res
def get_files(
    path,
    extensions=None,
    recurse=True,
    folders=None,
    followlinks=True,
):
    """
    Find all files in a folder recursively.
    Arguments:
        path: Path to input folder.
        extensions: list of acceptable file extensions.
        recurse: whether to perform a recursive search or not.
        folders: direct subfolders to explore (if None explore all).
        followlinks: whether to follow symlinks or not.
    Returns:
        List of all absolute paths to found files.
    """
    path = Path(path)
    folders = L(folders)
    extensions = setify(extensions)
    extensions = {e.lower() for e in extensions}
    if recurse:
        res = []
        for i, (p, d, f) in enumerate(
            os.walk(path, followlinks=followlinks)
        ):  # returns (dirpath, dirnames, filenames)
            if len(folders) != 0 and i == 0:
                d[:] = [o for o in d if o in folders]
            else:
                d[:] = [o for o in d if not o.startswith(".")]
            if len(folders) != 0 and i == 0 and "." not in folders:
                continue
            res += _get_files(p, f, extensions)
    else:
        f = [o.name for o in os.scandir(path) if o.is_file()]
        res = _get_files(path, f, extensions)
    return L(res)



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
    help="Size of the patches used foor training. Default 1024.",
)

parser.add_argument(
    "--area-threshold",
    type=int,
    default=50,
    help="Minimum area of objects to keep. Default 50.",
)

if __name__ == "__main__":
    print('hello !!')
    args = parser.parse_args()

    print(args)
    print(args.outfolder)
    print(args.mask_folder)

    mask_paths = get_files(args.mask_folder, extensions=".png", recurse=False).sorted(
        key=lambda x: x.stem
    )
    slide_name = mask_paths[0].stem.split("_")[0] + mask_paths[0].stem.split("_")[1]

    polygons = []
    for mask_path in mask_paths:
        patch_x = int(mask_paths[0].stem.split("_")[2])
        patch_y = int(mask_paths[0].stem.split("_")[3])
        print(mask_path)
        all_polygons = []
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        for sh, _ in features.shapes(mask.astype(np.int16), mask=(mask > 0)):
            all_polygons.append(shape(sh)         )
        all_polygons = MultiPolygon(all_polygons)

        polygon = translate(all_polygons, xoff=patch_x, yoff=patch_y)

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
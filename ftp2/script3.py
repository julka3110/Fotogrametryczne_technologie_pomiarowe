import laspy 
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS
import sys


def las_to_points(las):
    return np.vstack((las.x, las.y, las.z)).T

def points_to_raster(points, cell_size):
    min_x, min_y = points[:, 0].min(), points[:, 1].min()
    max_x, max_y = points[:, 0].max(), points[:, 1].max()

    n_cols = int(np.ceil((max_x - min_x) / cell_size))
    n_rows = int(np.ceil((max_y - min_y) / cell_size))

    lower_left = np.array([min_x, min_y])

    i = ((points[:, 0] - lower_left[0]) / cell_size).astype(int)
    j = ((points[:, 1] - lower_left[1]) / cell_size).astype(int)

    raster = np.full((n_rows, n_cols), np.nan)

    for col, row, z in zip(i, j, points[:, 2]):
        row = n_rows - 1 - row
        if np.isnan(raster[row, col]):
            raster[row, col] = z
        else:
            raster[row, col] = np.max([raster[row, col], z])

    transform = from_origin(lower_left[0], lower_left[1] + n_rows * cell_size, cell_size, cell_size)
    crs = CRS.from_epsg(2180) 

    return raster, transform, crs

def save_raster(raster,transform,crs,filepath):
    with rasterio.open(
        filepath,
        "w",
        driver="GTiff",
        height=raster.shape[0],
        width=raster.shape[1],
        count=1,
        dtype=rasterio.float32,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(raster, 1)


if __name__ == '__main__':   
    las1 = laspy.read(sys.argv[1])
    las2 = laspy.read(sys.argv[2])
    raster_wynikowy = sys.argv[3]
    
    points1 = las_to_points(las1)
    points2 = las_to_points(las2)

    nmt_points1=points1[las1.classification == 2]
    nmt_points2=points2[las2.classification == 2]
    nmpt1_points=points1[np.isin(las1.classification, [2,3,4,5,6])]
    nmpt2_points=points2[np.isin(las2.classification,[2,3,4,5,6])]


    nmt1,transform1,crs1= points_to_raster(nmt_points1, 1)
    save_raster(nmt1,transform1,crs1,"nmt1.tif")
    nmt2,transform2,crs2= points_to_raster(nmt_points2, 1)
    save_raster(nmt2,transform2,crs2,"nmt2.tif")
    nmpt1,transform_nmpt1,crs_nmpt1= points_to_raster(nmpt1_points, 1)
    save_raster(nmpt1,transform_nmpt1,crs_nmpt1,"nmpt1.tif")
    nmpt2,transform_nmpt2,crs_nmpt2= points_to_raster(nmpt2_points, 1)
    save_raster(nmpt2,transform_nmpt2,crs_nmpt2,"nmpt2.tif")
    nmpt_diff = nmpt1 - nmpt2
    save_raster(nmpt_diff,transform_nmpt1,crs_nmpt1, raster_wynikowy)
    


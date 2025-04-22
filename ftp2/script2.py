import laspy 
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import argparse
from scipy.spatial import cKDTree


def normalizacja_rgb(las):
    r = las.red / max(las.red)
    g = las.green / max(las.green)
    b = las.blue / max(las.blue)
    return r, g, b


def point_extraction_based_on_the_class(las, class_type):
    if class_type == 'buildings':
        #print('Ekstrakcja punktów budynków')
        # Klasyfikacja 6 oznacza budynki
        buildings_points = las.points[las.classification == 6]
        return buildings_points
    elif class_type == 'vegetation':
        #print('Ekstrakcja punktów roślinności')
        vegetation_low = las.points[las.classification == 3]
        vegetation_medium = las.points[las.classification == 4]
        vegetation_high = las.points[las.classification == 5]
        vegetation= las.points[(las.classification == 3) | (las.classification == 4) | (las.classification == 5)]
        return vegetation, vegetation_low, vegetation_medium, vegetation_high
    else:
        #print('Ekstrakcja punktów gruntu')
        # Klasyfikacja 2 oznacza grunt
        ground_points = las.points[las.classification == 2]
        return ground_points
    
    
def build_cloud(las):
    x, y, z = las.x, las.y, las.z
    points = np.vstack((x, y, z)).T
    r, g, b = normalizacja_rgb(las)
    las_points = np.vstack((x,y,z)).transpose()
    las_colors = np.vstack((r,g,b)).transpose()
    chmura_punktow = o3d.geometry.PointCloud()
    chmura_punktow.points = o3d.utility.Vector3dVector(las_points)
    #print(chmura_punktow.points)
    chmura_punktow.colors = o3d.utility.Vector3dVector(las_colors)
    return chmura_punktow


def calculate_density(las, radius, mode):
    pcd = build_cloud(las)
    points = np.asarray(pcd.points)

    tree = cKDTree(points)

    if mode == '2D':
        area = np.pi * radius**2
    elif mode == '3D':
        area = (4 / 3) * np.pi * radius**3
        
    print("Obliczanie gęstości...")
    densities = []
    for point in points:
        neighbors = tree.query_ball_point(point, radius)
        density = len(neighbors) / area 
        densities.append(density)
    
    return densities

    
def plot_density(densities,mode):   
    plt.hist (densities, bins = 30, color = 'blue', edgecolor = 'black') 
    plt.title('Histogram gęstości punktów')
    unit = 'm2' if mode == '2D' else 'm3'
    plt.xlabel(f'Gęstość punktów na {unit}')
    plt.ylabel('Liczba punktów')
    plt.grid(axis='y', alpha=0.75)
    plt.show()                  

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gęstość chmury punktów')
    parser.add_argument('las_path', type=str, help='Ścieżka do pliku las')
    parser.add_argument('--mode', type=str,choices = ['2D', '3D'], default='2D', help='Tryb obliczania gęstości')
    parser.add_argument('--ground_only', action='store_true', help='Oblicz gęstość tylko dla punktów gruntu')
    args = parser.parse_args()
    
    las = laspy.read(args.las_path)
    mode = args.mode
    if args.ground_only:
        las = point_extraction_based_on_the_class(las, 'ground')
    
    densities=calculate_density(las,1,mode)
    plot_density(densities,mode)
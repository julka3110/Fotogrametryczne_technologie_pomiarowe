import laspy 
import numpy as np
import open3d as o3d
import random
import os
import sys 






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
    las_points = np.vstack((x,y,z)).transpose()
    chmura_punktow = o3d.geometry.PointCloud()
    chmura_punktow.points = o3d.utility.Vector3dVector(las_points)
    return chmura_punktow

def calculate_eps(las):
    pcd = las
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    search_point = pcd.points[150]
    [k, idx, _] = pcd_tree.search_knn_vector_3d(search_point, 500)
    distances = []
    for i in idx:
        neighbor_point = np.asarray(pcd.points[i]) 
        distance = np.linalg.norm(search_point - neighbor_point)  
        distances.append(distance)

    eps = np.mean(distances)
    #print(f"Obliczona wartość eps: {eps}")
    return eps
    
def klasteryzacja_DBSCAN(chmura_punktow, odleglosc_miedzy_punktami, min_punktow, progress=True):
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        klasy = np.array(chmura_punktow.cluster_dbscan(eps=odleglosc_miedzy_punktami, min_points=min_punktow, print_progress=progress))
    liczba_klas = klasy.max() + 1
    print("Algorytm DBSCAN wykrył %i klas" % liczba_klas)
    colors = np.zeros((len(klasy), 3))  
    for i in range(liczba_klas):
        random_color = [random.random(), random.random(), random.random()] 
        colors[klasy == i] = random_color
    colors[klasy < 0] = [0.5, 0.5, 0.5]  
    chmura_punktow.colors = o3d.utility.Vector3dVector(colors)
    
    return chmura_punktow,klasy

def zapisz_budynki_do_laz(chmura_punktow, klasy, folder_docelowy="budynki_laz"):

    if not os.path.exists(folder_docelowy):
        os.makedirs(folder_docelowy)

    punkty = np.asarray(chmura_punktow.points)
    
    for klasa in range(klasy.max() + 1): 
        maska = (klasy == klasa) 
        punkty_budynku = punkty[maska]
        
        if len(punkty_budynku) > 0:
            las = laspy.LasData(laspy.header.LasHeader(point_format=3, version="1.2"))            
            las.x = punkty_budynku[:, 0]
            las.y = punkty_budynku[:, 1]
            las.z = punkty_budynku[:, 2]
            
            nazwa_pliku = os.path.join(folder_docelowy, f"budynek_{klasa}.laz")
            las.write(nazwa_pliku)
    print("Zapisano chmury punktów budynków do plików LAZ")



if __name__ == '__main__':    
    las = laspy.read(sys.argv[1])    
    folder_docelowy = sys.argv[2]       
    budynki = point_extraction_based_on_the_class(las, 'buildings')   
    chmura_budynkow = build_cloud(budynki)
    klastry_budynkow,klasy_budynkow = klasteryzacja_DBSCAN(chmura_budynkow, 3, 10)

    grunt = point_extraction_based_on_the_class(las, 'ground')
    chmura_gruntu = build_cloud(grunt)
    chmura_gruntu.paint_uniform_color([0.5, 0.3, 0.1]) 

    o3d.visualization.draw_geometries([klastry_budynkow, chmura_gruntu])
    zapisz_budynki_do_laz(klastry_budynkow, klasy_budynkow,folder_docelowy)



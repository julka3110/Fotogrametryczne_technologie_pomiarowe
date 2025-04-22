import laspy 
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import sys

colors = {
    0: [0.5, 0.5, 0.5],  
    1: [0.8, 0.8, 0.8],  
    2: [0.6, 0.4, 0.2],  
    3: [0.0, 0.4, 0.0],  
    4: [0.0, 0.8, 0.0],  
    5: [0.0, 1.0, 0.0],  
    6: [1.0, 0.0, 0.0], 
    7: [1.0, 1.0, 0.0],  
    8: [0.0, 1.0, 1.0],  
    9: [0.0, 0.0, 1.0], 
    10: [0.3, 0.3, 0.3], 
    11: [0.4, 0.4, 0.4], 
    12: [1.0, 0.5, 0.0], 
}

def normalizacja_rgb(las):# Normalizacja wartości RGB
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
    elif class_type == 'klasa0':
        points_0 = las.points[las.classification == 0]
        return points_0
    elif class_type == 'klasa1':
        points_1 = las.points[las.classification == 1]
        return points_1
    elif class_type == 'klasa7':
        points_7 = las.points[las.classification == 7]
        return points_7
    elif class_type == 'klasa8':
        points_8 = las.points[las.classification == 8]
        return points_8
    elif class_type == 'klasa9':
        points_9 = las.points[las.classification == 9]
        return points_9
    elif class_type == 'klasa10':
        points_10 = las.points[las.classification == 10]
        return points_10
    elif class_type == 'klasa11':
        points_11 = las.points[las.classification == 11]
        return points_11
    elif class_type == 'klasa12':
        points_12 = las.points[las.classification == 12]
        return points_12
    else:
        #print('Ekstrakcja punktów gruntu')
        # Klasyfikacja 2 oznacza grunt
        ground_points = las.points[las.classification == 2]
        return ground_points


def plot_bar_chart(las):
    klasa0 = point_extraction_based_on_the_class(las, 'klasa0')
    klasa1 = point_extraction_based_on_the_class(las, 'klasa1')
    klasa2= point_extraction_based_on_the_class(las, 'ground')
    _,klasa3,klasa4,klasa5 = point_extraction_based_on_the_class(las, 'vegetation')
    klasa6 = point_extraction_based_on_the_class(las, 'buildings')
    klasa7 = point_extraction_based_on_the_class(las, 'klasa7')
    klasa8 = point_extraction_based_on_the_class(las, 'klasa8')
    klasa9 = point_extraction_based_on_the_class(las, 'klasa9')
    klasa10 = point_extraction_based_on_the_class(las, 'klasa10')
    klasa11 = point_extraction_based_on_the_class(las, 'klasa11')
    klasa12 = point_extraction_based_on_the_class(las, 'klasa12')
    
    osY = np.array([len(klasa0), len(klasa1), len(klasa2), len(klasa3), len(klasa4), len(klasa5), len(klasa6), len(klasa7), len(klasa8), len(klasa9), len(klasa10), len(klasa11), len(klasa12)])
    osX = np.array([i for i in range(13)])
    plt.bar(osX, osY, tick_label=[str(i) for i in range(13)])
    plt.title('Liczba punktów w danej klasie')
    plt.xlabel('Klasa')
    plt.ylabel('Liczba punktów')
    plt.show()
    
def manualne_przycinanie_chmury_punktów(chmura_punktow):
    print("Manualne przycinanie chmury punktów")
    print("Etapy przetwarzania danych:")
    print(" (0) Manualne zdefiniowanie widoku poprzez obrót myszka lub:")
    print(" (0.1) Podwójne wciśnięcie klawisza X - zdefiniowanie widoku ortogonalnego względem osi X")
    print(" (0.2) Podwójne wciśnięcie klawisza Y - zdefiniowanie widoku  ortogonalnego względem osi Y")
    print(" (0.3) Podwójne wciśnięcie klawisza Z - zdefiniowanie widoku ortogonalnego względem osi Z")
    print(" (1) Wciśnięcie klawisza K - zmiana na tryb rysowania")
    print(" (2.1) Wybór zaznaczenia poprzez wciśnięcie lewego przycisku myszy i interaktywnego narysowania prostokąta lub")
    print(" (2.2) przytrzymanie przycisku ctrl i wybór wierzchołków poligonu lewym przyciskiem myszy")
    print(" (3) Wciśnięcie klawisza C - wybór zaznaczonego fragmentu chmury punktów i zapis do pliku")
    print(" (4) Wciśnięcie klawisza F - powrót do interaktywnego wyświetlania chmury punktów")
    o3d.visualization.draw_geometries_with_editing([chmura_punktow],window_name='Przycinanie chmury punktów')

def visualize_cloud(las):
    print("Przygotowywanie wizualizacji chmury punktów...")
    x, y, z = las.x, las.y, las.z
    las_points = np.vstack((x,y,z)).transpose()
    las_colors = np.zeros((len(las_points), 3))
    for class_id, color in colors.items():
        las_colors[las.classification == class_id] = color
    chmura_punktow = o3d.geometry.PointCloud()
    chmura_punktow.points = o3d.utility.Vector3dVector(las_points)
    chmura_punktow.colors = o3d.utility.Vector3dVector(las_colors)
    manualne_przycinanie_chmury_punktów(chmura_punktow)


if __name__ == '__main__':
    las = laspy.read(sys.argv[1])
    plot_bar_chart(las)
    visualize_cloud(las)
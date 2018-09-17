# Kernel-Density

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import *
from math import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.grid_search import GridSearchCV

def kde(MoBike_file_path):
    
    T =[]
    start=[]
    stop=[]
    start_bbox=[]
    stop_bbox=[]

    csvfile=open(MoBike_file_path,'r',encoding='utf-8')
    lines=csv.reader(csvfile) #Attention : il faut encoder le fichier csv en 'utf-8'
    for line in lines:
        T.append(line)
        
   
    for k in range(1,len(T)):
        head_k=(T[k][0]).split(',')
        del head_k[9:]
        try :
            Debut_k=(float(head_k[4]), float(head_k[5]))
        except ValueError:
            print(head_k[4],head_k[5])
            pass
        start.append(Debut_k)
    
        try :
            end_k=(float(head_k[-2]),float(head_k[-1]))
        except ValueError:
            print(head_k[-2], head_k[-1])
            pass
        stop.append(end_k)
    

#Création de la BBox à partir d'un polygone créé sur ArcGIS

    r=shapefile.Reader('/home/mifsud-couchaux/Documents/drive-download-20180710T045435Z-001/Export_Output_Bounding_Box.shp')
    shapes=r.shapes()
    l=len(shapes) #l=1, il n'y a qu'un polygone dans ce fichier 
    bounding_box=shapes[0].points[:] #tableau contenant tous les points que forme le polygone 
    bbox=Polygon(batchConvertCoordinates(bounding_box,3395,4326)) #Créer un polygone (format shapely) à partir d'une liste de points

    for i in start: # a partir d'une liste de couples en gcj (ligne_k ici), ajoute des couples au format WGS84 à un fichier csv
        i_shifted=(gcj02_to_wgs84(i[0],i[1]))
        i_shifted_type_couple=gcj02_to_wgs84(i[0],i[1])[0],gcj02_to_wgs84(i[0],i[1])[1]
        if bbox.contains(Point(i_shifted_type_couple))==True: #ne récupère que les point de départ et d'arrivées inclus dans l'Inner RIng
            start_bbox.append(i_shifted_type_couple)
            
    for i in stop: # a partir d'une liste de couples en gcj (ligne_k ici), ajoute des couples au format WGS84 à un fichier csv
        i_shifted=(gcj02_to_wgs84(i[0],i[1]))
        i_shifted_type_couple=gcj02_to_wgs84(i[0],i[1])[0],gcj02_to_wgs84(i[0],i[1])[1]
        if bbox.contains(Point(i_shifted_type_couple))==True: #ne récupère que les point de départ et d'arrivées inclus dans l'Inner RIng
            stop_bbox.append(i_shifted_type_couple)
        
#NB : Polygon.contains(Point()) permet de vérifier qu'un couple est inclus dans un polygone ici l'Inner Ring
#Cela nous permet de déselectionner les points de départ ou d'arrivées localisés au même endroit que les centres 
# de recherche et de développement 
        
    
    start_epsg4479=batchConvertCoordinates(start_bbox,4326,4479) #Type depart=une liste de couples
    stop_epsg4479=batchConvertCoordinates(stop_bbox,4326,4479)


    start_epsg4479_array_type=np.array(start_epsg4479) #transforme les listes de couples en tableau de type array
    stop_epsg4479_array_type=np.array(stop_epsg4479)
    
    print(start_epsg4479_array_type)

    m1=start_epsg4479_array_type[:,0]
    m2=start_epsg4479_array_type[:,1]

    xmin = min(m1)
    xmax = max(m1)
    ymax = min(m2)
    ymin = max(m2)

    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j] 
    sample = np.vstack([Y.ravel(), X.ravel()]).T
    train= np.vstack([m2, m1]).T
    kernel=KernelDensity(kernel='gaussian',bandwidth=466.6666666).fit(train) #training 
    Z = np.reshape(np.exp(kernel.score_samples(sample)),X.shape) #learning


    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.imshow(np.rot90(Z), cmap=cm.coolwarm,extent=[xmin, xmax, ymin, ymax])
    #ax[1].plot(m1, m2, 'k.', markersize=0.5)
    ax.set_title('Gaussian KDE, MoBike starting points in Shanghai - 5/7/2018')
    #ax[1].set_title('MoBike starting points scatter - 5/7/2018')
    fig.set_size_inches(20,20)
    plt.gca().set_aspect('equal')
    plt.gca().set_xlim(xmin,xmax)
    plt.gca().set_ylim(ymin,ymax)

    Repères_visuels=[(121.450446,31.251552),(121.495,31.242),(121.440833,31.225),(121.431389,31.193056),(121.498144,31.200808),(121.473056,31.232222)]
    Names=['SRS','Oriental Pearl','JA Temple','St Ignacius Cath','Shanghai MOCA','People Park']

    References_euclidian=batchConvertCoordinates(Repères_visuels,4326,4479)
    References_euclidian=np.array(References_euclidian)
    
    for index, i in enumerate(References_euclidian):
        plt.text(i[0],i[1],Names[index], color='red')
        plt.scatter(i[0],i[1],s=20)
        
    plt.show()
    
    m1=stop_epsg4479_array_type[:,0]
    m2=stop_epsg4479_array_type[:,1]
    
    xmin = min(m1)
    xmax = max(m1)
    ymax = min(m2)
    ymin = max(m2)

    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j] 
    sample = np.vstack([Y.ravel(), X.ravel()]).T
    train= np.vstack([m2, m1]).T
    kernel=KernelDensity(kernel='gaussian',bandwidth=466.6666666).fit(train) #training 
    Z = np.reshape(np.exp(kernel.score_samples(sample)),X.shape) #learning


    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.imshow(np.rot90(Z), cmap=cm.coolwarm,extent=[xmin, xmax, ymin, ymax])
    #ax[1].plot(m1, m2, 'k.', markersize=0.5)
    ax.set_title('Gaussian KDE, MoBike starting points in Shanghai - 5/7/2018')
    #ax[1].set_title('MoBike starting points scatter - 5/7/2018')
    fig.set_size_inches(20,20)
    plt.gca().set_aspect('equal')
    plt.gca().set_xlim(xmin,xmax)
    plt.gca().set_ylim(ymin,ymax)

    
        
    plt.show()

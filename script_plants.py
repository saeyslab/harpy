# %load_ext autoreload
# %autoreload 2
import pandas as pd
import numpy as np
from anndata import AnnData
import squidpy as sq
import scanpy as sc
from cellpose import models, io
import matplotlib.pyplot as plt
from modest_image import ModestImage, imshow
from skimage.restoration import inpaint
#%matplotlib widget
import cv2
import torch
from skimage import io, morphology
import skimage.feature as features
from scipy import ndimage
import time
from tqdm.notebook import tqdm
import scipy
import seaborn as sns
import pybasic


import skimage
from skimage.measure import regionprops, regionprops_table, approximate_polygon
from shapely.geometry import Polygon
from shapely.geometry.polygon import LinearRing
import geopandas
import scanpy as sc
import rasterio
from rasterio import features
import shapely
from shapely.geometry import Point, Polygon
import matplotlib
from matplotlib.colors import Normalize


def BasiCCorrection(path_image=None,I=None):
    
    if I is None:
        I = io.imread(path_image)
    #create the tiles
    
    Tiles=[]
    for i in range(0,int(I.shape[0]/2144)): #over the rows
        for j in range(0,int(I.shape[1]/2144)):
            Temp=I[i*2144:(i+1)*2144,j*2144:(j+1)*2144]
            Tiles.append(Temp)
     #measure the filters       
    flatfield = pybasic.basic(Tiles, darkfield=False,verbosity=False)
    
    tiles_corrected= pybasic.correct_illumination(images_list = Tiles, flatfield = flatfield[0])#, darkfield = darkfield)
    
    # stitch the tiles back together
    Inew=np.zeros(I.shape)
    k=0
    for i in range(0,int(I.shape[0]/2144)): #over the rows
        for j in range(0,int(I.shape[1]/2144)):
                Inew[i*2144:(i+1)*2144,j*2144:(j+1)*2144]=tiles_corrected[k]
                k=k+1
      
    fig,ax =plt.subplots(1,2,figsize=(20,10))
    ax[0].imshow(Inew,cmap='gray')
    ax[1].imshow(I,cmap='gray')
    
    return Inew.astype(np.uint16)

def preprocessImage(I=None,path_image=None,contrast_clip=2.5,size_tophat=None,small_size_vis=None):
    "This function performs the prprocessing of an image. If the path_image i provided, the image is read from the path." 
    "If the image I itself is provided, this image will be used."
    "Contrast_clip indiactes the input to the create_CLAHE function for histogram equalization"
    "size_tophat indicates the tophat filter size. If no tophat lfiter size is given, no tophat filter is executes. The recommendable size is 45?-."
    "Small_size_vis indicates the coordinates of an optional zoom in plot to check the processing better."
    t0=time.time()
    #Read in image 
    if I is None:
        I = io.imread(path_image)
    Iorig=I
    
    #mask black lines 
    maskLines=np.where(I == 0) # find the location of the lines 
    mask=np.zeros(I.shape,dtype=np.uint8)
    mask[maskLines[0],maskLines[1]]=1 # put one values in the correct position
    
    # perform inpainting
    res_NS = cv2.inpaint(I, mask,55, cv2.INPAINT_NS)
    I=res_NS
    
    #tophat filter
    if size_tophat is not None:
        minimum_t = ndimage.minimum_filter(I, size_tophat)  
        orig_sub_min = I - minimum_t
        I=orig_sub_min
    
    #enhance contrast
    clahe = cv2.createCLAHE(clipLimit=contrast_clip, tileGridSize=(8,8))
    I = clahe.apply(I)
    
    #plot_result
    fig,ax =plt.subplots(1,2,figsize=(20,10))
    ax[0].imshow(Iorig,cmap='gray')
    ax[1].imshow(I,cmap='gray')
    
    #plot small part of image
    if small_size_vis is not None:
        fig,ax =plt.subplots(1,2,figsize=(20,10))
        ax[0].imshow(Iorig[small_size_vis[0]:small_size_vis[1],small_size_vis[2]:small_size_vis[3]],cmap='gray')
        ax[1].imshow(I[small_size_vis[0]:small_size_vis[1],small_size_vis[2]:small_size_vis[3]],cmap='gray')
    print(time.time()-t0)    
    return I


def segmentation(I,device='cpu',min_size=80,flow_threshold=0.6,diameter=55,mask_threshold=0,small_size_vis=None,model_type='nuclei',channels=np.array([0,0])):
    "This function segments the data, using the cellpose algorithm, and plots the outcome"
    "I is the input image, showing the DAPI Staining, you can define your device by setting the device parameter"
    "min_size indicates the minimal amount of pixels in a mask (I assume)"
    "The lfow_threshold indicates someting about the shape of the masks, if you increase it, more masks with less orund shapes will be accepted"
    "The diameter is a very important parameter to estimate. In the best case, you estimate it yourself, it indicates the mean expected diameter of your dataset."
    "If you put None in diameter, them odel will estimate is herself."
    "mask_threshold indicates how many of the possible masks are kept. MAking it smaller (up to -6), will give you more masks, bigger is less masks. "
    "When an RGB image is given a input, the R channel is expected to have the nuclei, and the blue channel the membranes"
    "When whole cell segmentation needs to be performed, model_type=cyto, otherwise, model_type=nuclei"
    t0=time.time()
    device = torch.device(device) #GPU 4 is your GPU
    torch.cuda.set_device(device)
    model = models.Cellpose(gpu=device, model_type=model_type)
    print(torch.cuda.current_device())
    channels=channels
    torch.cuda.set_device(device)
    masks, flows, styles, diams = model.eval(I,diameter=diameter,channels=channels,min_size=min_size,flow_threshold=flow_threshold,mask_threshold=mask_threshold)
    masksI = np.ma.masked_where(masks == 0, masks)
    Imasked=np.ma.masked_where(I < 500 , I)
     # create the polygon shapes of the different cells 
    polygons=mask_to_polygons_layer(masks)
    #polygons["border"] = polygons.geometry.map(is_in_border)
    polygons["border_color"] = polygons.geometry.map(border_color)
    polygons["linewidth"] = polygons.geometry.map(linewidth)
    polygons["color"] = polygons.geometry.map(color)
    polygons['cells']=polygons.index

    polygons=polygons.dissolve(by='cells')

    

    #visualization
    if sum(channels)!=0:
        I=I[0,:,:] # select correct image
    

    if small_size_vis is not None:
        small=polygons.cx[small_size_vis[0]:small_size_vis[1],small_size_vis[2]:small_size_vis[3]]
        fig,ax =plt.subplots(1,2,figsize=(20,10))
        ax[0].imshow(I[small_size_vis[0]:small_size_vis[1],small_size_vis[2]:small_size_vis[3]],cmap='gray')

        ax[1].imshow(I[small_size_vis[0]:small_size_vis[1],small_size_vis[2]:small_size_vis[3]],cmap='gray')
        small.plot(ax=ax[1],edgecolor='white',linewidth=small.linewidth,alpha=0.5,legend=True,color='red')
        plt.show()
        #fig,ax =plt.subplots(1,2,figsize=(20,10))
        #ax[0].imshow(I[small_size_vis[0]:small_size_vis[1],small_size_vis[2]:small_size_vis[3]],cmap='gray')

        #ax[1].imshow(I[small_size_vis[0]:small_size_vis[1],small_size_vis[2]:small_size_vis[3]],cmap='gray')
        #ax[1].imshow(masksI[small_size_vis[0]:small_size_vis[1],small_size_vis[2]:small_size_vis[3]],cmap='jet')
        #plt.show()
    else: 
        fig,ax =plt.subplots(1,2,figsize=(20,10))
        #ax[0].imshow(masks[0:3000,8000:10000],cmap='jet')
        ax[0].imshow(I,cmap='gray')
        #ax[0].imshow(masks,cmap='jet')

        ax[1].imshow(I,cmap='gray')
        polygons.plot(ax=ax[1],edgecolor='white',linewidth=polygons.linewidth,alpha=0.5,legend=True,color='red')
        #ax[1].imshow(masksI,cmap='jet')
        plt.show()
    print(time.time()-t0)
    return masks   

def mask_to_polygons_layer(mask):
    # https://rocreguant.com/convert-a-mask-into-a-polygon-for-images-using-shapely-and-rasterio/1786/
    all_polygons = []
    all_values = []
    for shape, value in features.shapes(mask.astype(np.int16), mask=(mask >0), transform=rasterio.Affine(1.0, 0, 0, 0, 1.0, 0)):
        all_polygons.append(shapely.geometry.shape(shape))
        all_values.append(int(value))

    return geopandas.GeoDataFrame(dict(geometry=all_polygons), index=all_values)



def color(r):
    return plt.get_cmap("Set1")(np.random.choice(np.arange(0, 18)))

def border_color(r):
    return plt.get_cmap("tab10")(3) if r else (1, 1, 1 ,1)

def linewidth(r):
    return 1 if r else 0.5

def is_in_border(r):
    r = r.centroid
    if (r.x - border_margin < 0) or (r.x + border_margin > h):
        return True
    if (r.y - border_margin < 0) or (r.y + border_margin > w):
        return True
    return False


def create_adata_quick(path,I,Masks,library_id='melanoma'):

    # create the polygon shapes of the different cells 
    polygons=mask_to_polygons_layer(Masks)
    #polygons["border"] = polygons.geometry.map(is_in_border)
    polygons["border_color"] = polygons.geometry.map(border_color)
    polygons["linewidth"] = polygons.geometry.map(linewidth)
    polygons["color"] = polygons.geometry.map(color)
    polygons['cells']=polygons.index

    polygons=polygons.dissolve(by='cells')

    fig,ax =plt.subplots(1,2,figsize=(20,10))
    ax[0].imshow(I,cmap='gray')
    polygons.plot(ax=ax[0],column='color',edgecolor='white',linewidth=polygons.linewidth,alpha=0.5)
    ax[1].imshow(I,cmap='gray')
    # allocate the transcripts 
    df = pd.read_csv(path, delimiter = "\t",header=None)
    df=df[(df[1]<Masks.shape[0])&(df[0]<Masks.shape[1])]
    df['cells']=Masks[df[1].values,df[0].values]    
    coordinates=df.groupby(['cells']).mean().iloc[:,[0,1]] #calculate the mean of the transcripts for every cell. Now based on transcripts, better on masks? 
    #based on masks is present in the adata.obsm

    # create the anndata object 
    cellCounts=df.groupby(['cells',3]).size().unstack(fill_value=0) #create a matrix based on counts 
    adata = AnnData(cellCounts[cellCounts.index!=0])
    coordinates.index=coordinates.index.map(str)
    adata.obsm['spatial'] = coordinates[coordinates.index!='0']

    # add the polygons to the anndata object
    polygonsF=polygons[np.isin(polygons.index.values,list(map(int,adata.obs.index.values)))]
    polygonsF.index=list(map(str,polygonsF.index))
    adata.obsm['polygons']=polygonsF

    # add the figure to the anndata
    spatial_key = "spatial"
    library_id = library_id
    adata.uns[spatial_key] = {library_id: {}}
    adata.uns[spatial_key][library_id]["images"] = {}
    adata.uns[spatial_key][library_id]["images"] = {"hires": I}
    adata.uns[spatial_key][library_id]["scalefactors"] = {"tissue_hires_scalef": 1, "spot_diameter_fullres": 75}


    return adata


def plot_shapes(adata,column=None,cmap='magma',alpha=0.5,crd=None):
    # This function plots the anndata on the shapes of the cells, but it does not do it smartly.
    if column is not None:
        
        if column+'_colors'  in adata.uns:
            print('using the colormap defined in the anndata object')
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list('new_map',adata.uns[column+'_colors'], N=len(adata.uns[column+'_colors']))

        fig,ax =plt.subplots(1,1,figsize=(20,20))
        ax.imshow(adata.uns['spatial']['melanoma']['images']['hires'],cmap='gray',)
        adata.obsm['polygons'].plot(ax=ax,column=adata.obs[column],edgecolor='white',linewidth=adata.obsm['polygons'].linewidth,alpha=alpha,legend=True)
    else: 
        fig,ax =plt.subplots(1,1,figsize=(20,20))
        ax.imshow(adata.uns['spatial']['melanoma']['images']['hires'],cmap='gray',)
        adata.obsm['polygons'].plot(ax=ax,edgecolor='white',linewidth=adata.obsm['polygons'].linewidth,alpha=alpha,legend=True,color='blue')
        
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    #ax.legend(bbox_to_anchor=(1.1, 1.05))
    if crd is not None:
        ax.set_xlim(crd[0], crd[1])
        ax.set_ylim(crd[2], crd[3])
    #ax[1].imshow(I,cmap='gray',)
    
def preprocessAdata(adata,mask,Nuc_size_norm=True):
    

    sc.pp.calculate_qc_metrics(adata, inplace=True,percent_top=[2,5])
    fig, axs = plt.subplots(1, 2, figsize=(15, 4))
    sns.distplot(adata.obs["total_counts"], kde=False,ax=axs[0])
    sns.distplot(adata.obs["n_genes_by_counts"], kde=False, bins=55,ax=axs[1])
    sc.pp.filter_cells(adata, min_counts=10) 
    sc.pp.filter_genes(adata, min_cells=5) 
    adata.raw = adata
    
    # nucleusSizeNormalization
    if Nuc_size_norm == True :
        unique, counts = np.unique(mask, return_counts=True)
        nucleusSize=[]
        for i in adata.obs.index:
            nucleusSize.append(counts[int(i)])
        adata.obs['nucleusSize']=nucleusSize 
        adata.X=(adata.X.T/adata.obs.nucleusSize.values).T 
        
        #sc.pp.normalize_total(adata) #This no
        sc.pp.log1p(adata)
        sc.pp.scale(adata, max_value=10)
    else:
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
    sc.tl.pca(adata, svd_solver='arpack',n_comps=50)
    sc.pl.pca(adata, color='total_counts')  
    sc.pl.pca_variance_ratio(adata,n_pcs=50) #lets take 6,10 or 12
    adata.obsm['polygons']=geopandas.GeoDataFrame(adata.obsm['polygons'],geometry=adata.obsm['polygons'].geometry)
    plt.scatter(adata.obs['nucleusSize'],adata.obs['total_counts'])
    plt.title=('cellsize vs cellcount')
    return adata


def filter_on_size(adata,min_size=100,max_size=100000):
    start=adata.shape[0]
    adata.obsm['polygons']["X"] = adata.obsm['polygons'].centroid.x
    adata.obsm['polygons']["Y"] = adata.obsm['polygons'].centroid.y
    adata.obs['distance']=np.sqrt(np.square(adata.obsm['polygons']["X"]-adata.obsm['spatial'][0])+np.square(adata.obsm['polygons']["Y"]-adata.obsm['spatial'][1]))
    
    adata=adata[adata.obs['nucleusSize']<max_size,:]
    adata=adata[adata.obs['nucleusSize']>min_size,:]
    adata=adata[adata.obs['distance']<70,:]

    adata.obsm['polygons']=geopandas.GeoDataFrame(adata.obsm['polygons'],geometry=adata.obsm['polygons'].geometry)
    filtered=start-adata.shape[0]
    plot_shapes(adata,column='distance')
    print(str(filtered)+' cells were filtered out based on size.')
    return adata

def preprocess3(adata,pcs,neighbors,spot_size=70):
    
    sc.pp.neighbors(adata, n_neighbors=neighbors, n_pcs=pcs)
    sc.tl.umap(adata)
    #sc.pl.umap(adata, color=['Folr2','Glul','Sox9','Cd9']) #total counts doesn't matter that much
    sc.tl.leiden(adata,resolution=1)
    sc.pl.umap(adata, color=['leiden'])
    sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')
    sc.pl.rank_genes_groups(adata, n_genes=8, sharey=False)
    
        
    plot_shapes(adata,column='leiden')
    return adata


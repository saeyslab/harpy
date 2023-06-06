""" This file holds all the general functions that are used to build up the pipeline and the notebooks. The functions are odered by their occurence in the pipeline from top to bottom."""

# %load_ext autoreload
# %autoreload 2
import warnings
from itertools import chain
from typing import List, Optional, Tuple

import cv2
import os
os.environ['USE_PYGEOS'] = '0'
import geopandas
import matplotlib
import matplotlib.colors as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import scanpy as sc
import scipy
import seaborn as sns
import shapely
import squidpy as sq
import torch
from anndata import AnnData
from basicpy import BaSiC
from cellpose import models
from rasterio import features
import scipy as sp
from scipy import ndimage
from scipy.ndimage import gaussian_filter
import dask.dataframe as dd
import spatialdata
import xarray as xr
import dask
import numpy as np
import hvplot.xarray
import numpy as np
from longsgis import voronoiDiagram4plg
from shapely.geometry import Polygon
from shapely.geometry.polygon import LinearRing
from scipy.ndimage.filters import gaussian_filter



def read_in_zarr(path_to_zarr_file,zyx_order=[0,1,2],subset=None):
    """This function read in a zarr file containing the tissue image. If a z-stack is present, automatically, a z-projection is performed.
     A quidpy imagecontainer is returned."""
    da = dask.array.from_zarr(path_to_zarr_file)
    da = xr.DataArray(
    da, 
    dims=('z', 'y', 'x'),
    coords={
        "z": np.arange(da.shape[zyx_order[0]]),
        "y": np.arange(da.shape[zyx_order[1]]), 
        "x": np.arange(da.shape[zyx_order[2]]),
    }
    )
    if da.z.shape[0]>1:
        da=da.max(dim='z')
    if subset:    
        da=da[subset[0]:subset[1],subset[2]:subset[3]]    
    ic = sq.im.ImageContainer(da,layer='raw_image')    

    return ic

def create_subset_image(ic,crd):
    """Reads in sq image container and returns a subset in a sq.imagecontainer"""
    Xmax=ic.data.sizes['x']
    Ymax=ic.data.sizes['y']    
    img=ic.data.assign_coords({'x':np.arange(Xmax),'y':np.arange(Ymax)})  
    I_small=img['image'].sel(x=slice(crd[0],crd[1]),y=slice(crd[2],crd[3]))
    ic_test= sq.im.ImageContainer(I_small)
    return ic_test


def tilingCorrection(
    img: sq.im.ImageContainer,
    left_corner: Tuple[int, int] = None,
    size: Tuple[int, int] = None,
    tile_size: int = 2144,
) -> Tuple[sq.im.ImageContainer, np.ndarray]:
    """Returns the corrected image and the flatfield array

    This function corrects for the tiling effect that occurs in some image data for example the resolve dataset.
    The illumination within the tiles is adjusted, afterwards the tiles are connected as a whole image by inpainting the lines between the tiles.
    """

    # Create the tiles
    tiles = img.generate_equal_crops(size=tile_size, as_array="raw_image")
    tiles = np.array([tile + 1 if ~np.any(tile) else tile for tile in tiles])
    black=np.array([1 if ~np.any(tile-1) else 0 for tile in tiles]) # determine if 

    #create the masks for inpainting 
    i_mask = (np.block(
        [
            list(tiles[i : i + (img.shape[1] // tile_size)])
            for i in range(0, len(tiles), img.shape[1] // tile_size)
        ]
    ).astype(np.uint16)==0)
    if tiles.shape[0]<5:
        print('There aren\'t enough tiles to perform tiling correction (less than 5). This step will be skipped.')
        tiles_corrected=tiles
        flatfield=None
    else:
        basic = BaSiC(smoothness_flatfield=1)
        basic.fit(tiles)
        flatfield = basic.flatfield
        tiles_corrected = basic.transform(tiles)


    tiles_corrected = np.array(
        [tiles[number] if black[number]==1 else tile for number,tile in enumerate(tiles_corrected)]
    )

    # Stitch the tiles back together
    i_new = np.block(
        [
            list(tiles_corrected[i : i + (img.shape[1] // tile_size)])
            for i in range(0, len(tiles_corrected), img.shape[1] // tile_size)
        ]
    ).astype(np.uint16)

    

    mask = sq.im.ImageContainer(i_mask.astype(np.uint8), layer="mask_black_lines")
    #img.add_img(
     #   i_mask.astype(np.uint8),
     #   layer="mask",
    #)

    if size is not None and left_corner is not None:
        img = img.crop_corner(*left_corner, size)

    # Perform inpainting
    img.apply(
        {"0": cv2.inpaint},
        layer="raw_image",
        drop=False,
        channel=0,
        new_layer='corrected',
        copy=False,
        #chunks=10,
        fn_kwargs={
            "inpaintMask": mask.data.mask_black_lines.squeeze().to_numpy(),
            "inpaintRadius": 55,
            "flags": cv2.INPAINT_NS,
        },
    )

    print(img)
    return img, flatfield


def tilingCorrectionPlot(
    img: np.ndarray, flatfield, img_orig: np.ndarray, output: str = None
) -> None:
    """Creates the plots based on the correction overlay and the original and corrected images."""

    # disable interactive mode
    if output:
        plt.ioff()

    # Tile correction overlay
    if flatfield is not None:
        fig1, ax1 = plt.subplots(1, 1, figsize=(20, 10))
        ax1.imshow(flatfield, cmap="gray")
        ax1.set_title("Correction performed per tile")

        # Save the plot to ouput
        if output:
            plt.close(fig1)
            fig1.savefig(output + "0.png")

    # Original and corrected image
    fig2, ax2 = plt.subplots(1, 2, figsize=(20, 10))
    ax2[0].imshow(img, cmap="gray")
    ax2[0].set_title("Corrected image")
    ax2[1].imshow(img_orig, cmap="gray")
    ax2[1].set_title("Original image")
    ax2[0].spines["top"].set_visible(False)
    ax2[0].spines["right"].set_visible(False)
    ax2[0].spines["bottom"].set_visible(False)
    ax2[0].spines["left"].set_visible(False)
    ax2[1].spines["top"].set_visible(False)
    ax2[1].spines["right"].set_visible(False)
    ax2[1].spines["bottom"].set_visible(False)
    ax2[1].spines["left"].set_visible(False)
    
    # Save the plot to ouput
    if output:
        plt.close(fig2)
        fig2.savefig(output + "1.png")


def preprocessImage(
    img: sq.im.ImageContainer,
    img_layer=None,
    contrast_clip: float = 2.5,
    size_tophat: int = None,
) -> sq.im.ImageContainer:
    """Returns the new image

    This function performs the preprocessing of the image.
    Contrast_clip indicates the input to the create_CLAHE function for histogram equalization.
    Size_tophat indicates the tophat filter size. If no tophat lfiter size is given, no tophat filter is applied. The recommendable size is 45.
    Small_size_vis indicates the coordinates of an optional zoom in plot to check the processing better.
    """
    if img_layer==None:
        if 'corrected' in img:
            img_layer='corrected'
        elif 'raw_image' in img:
            img_layer='raw_image'
        else:
            raise ValueError('Your layers in you image container don\'t have the expected names. Please provide a correct name.')

    # Apply tophat filter
    if size_tophat is not None:
        minimum_t = ndimage.minimum_filter(
            img.data[img_layer].squeeze().to_numpy(), size_tophat
        )
        max_of_min_t = ndimage.maximum_filter(minimum_t, size_tophat)
        img.apply(
            {"0": lambda array: array - max_of_min_t},
            layer=img_layer,
            new_layer='corrected',
            drop=True,
            channel=0,
            copy=False,
        )
    if 'corrected' in img:
        img_layer='corrected'
    # Enhance the contrast
    clahe = cv2.createCLAHE(clipLimit=contrast_clip, tileGridSize=(8, 8))
    img.apply({"0": clahe.apply}, layer=img_layer,new_layer='corrected', drop=True, channel=0, copy=False)

    return img


def preprocessImagePlot(
    img: np.ndarray,
    img_orig: np.ndarray,
    crd: List[int] = None,
    output: str = None,
) -> None:
    """Creates the plots based on the original and preprocessed image."""

    # disable interactive mode
    if output:
        plt.ioff()

    # Original and preprocessed image
    fig1, ax1 = plt.subplots(1, 2, figsize=(20, 10))
    ax1[0].imshow(img, cmap="gray")
    ax1[0].set_title("Corrected image")
    ax1[1].imshow(img_orig, cmap="gray")
    ax1[1].set_title("Original image")

    # Save the plot to ouput
    if output:
        plt.close(fig1)
        fig1.savefig(output + "0.png")

    # Plot small part of the images
    if crd is not None:
        fig2, ax2 = plt.subplots(1, 2, figsize=(20, 10))
        ax2[0].imshow(
            img[
                crd[0] : crd[1],
                crd[2] : crd[3],
            ],
            cmap="gray",
        )
        ax2[0].set_title("Corrected image")
        ax2[1].imshow(
            img_orig[
                crd[0] : crd[1],
                crd[2] : crd[3],
            ],
            cmap="gray",
        )
        ax2[1].set_title("Original image")

        # Save the plot to ouput
        if output:
            plt.close(fig2)
            fig2.savefig(output + "1.png")

def cellpose(img, min_size=80,cellprob_threshold=-4, flow_threshold=0.85, diameter=100, model_type="cyto",channels=[1,0],device='cpu'):
    gpu=torch.cuda.is_available()
    model = models.Cellpose(gpu=gpu,model_type=model_type,device=torch.device(device))
    masks, _, _, _ = model.eval(
        img,
        diameter=diameter,
        channels=channels,
        min_size=min_size,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
    )
    return masks   
            
def segmentation(
    img: sq.im.ImageContainer,
    img_layer='corrected',
    device: str = "cpu",
    min_size: int = 80,
    flow_threshold: float = 0.6,
    diameter: int = 55,
    cellprob_threshold: int = 0,
    model_type: str = "nuclei",
    channels =[0, 0],
    ): 

    
    sq.im.segment(img=img, layer=img_layer, method=cellpose,chunks='auto',min_size=min_size,layer_added='segmentation_mask',cellprob_threshold=cellprob_threshold, flow_threshold=flow_threshold, diameter=diameter, model_type=model_type,channels=channels,device=device)
    masks=img.data.segmentation_mask.squeeze().to_numpy()
    #mask_i = np.ma.masked_where(masks == 0, masks)

    # Create the polygon shapes of the different cells
    polygons = mask_to_polygons_layer(masks)
    #polygons["border_color"] = polygons.geometry.map(fc.border_color)
    polygons["linewidth"] = polygons.geometry.map(linewidth)
    #polygons["color"] = polygons.geometry.map(fc.color)
    polygons["cells"] = polygons.index
    polygons.index=polygons.index.astype(str)
    polygons = polygons.dissolve(by="cells")
    sdata=imageContainerToSData(img)
    if model_type=='nuclei':
        sdata.add_shapes(name='nucleus_boundaries',shapes=spatialdata.models.ShapesModel.parse(polygons))
    elif model_type=='cyto':
        sdata.add_shapes(name='cell_boundaries',shapes=spatialdata.models.ShapesModel.parse(polygons))
    return sdata     


def imageContainerToSData(ic,sdata=None,layers_im=['corrected','raw_image'],layers_labels=['segmentation_mask']):
    
    if sdata==None:
        
        sdata=spatialdata.SpatialData()
        
    temp=ic.data.rename_dims({'channels':'c'})
    for i in layers_im:
       
        sdata.add_image(name=i,image=spatialdata.models.Image2DModel.parse(temp[i].squeeze(dim='z').transpose('c','y','x')))
    for j in layers_labels:
        sdata.add_labels(name=j,labels=spatialdata.models.Labels2DModel.parse(temp[j].squeeze().transpose('y','x')))
    
    return sdata             

def segmentationDeprecated(
    img: sq.im.ImageContainer,
    device: str = "cpu",
    min_size: int = 80,
    flow_threshold: float = 0.6,
    diameter: int = 55,
    cellprob_threshold: int = 0,
    model_type: str = "nuclei",
    channels: List[int] = [0, 0],
) -> Tuple[np.ndarray, np.ndarray, geopandas.GeoDataFrame, sq.im.ImageContainer]:
    """Returns the segmentation masks, the image masks and the polygons

    This function segments the data, using the cellpose algorithm.
    Img is the input image.
    You can define your device by setting the device parameter.
    Min_size indicates the minimal amount of pixels in a mask.
    The flow_threshold indicates someting about the shape of the masks, if you increase it, more masks with less orund shapes will be accepted.
    The diameter is a very important parameter to estimate, in the best case you estimate it yourself. It indicates the mean expected diameter of your dataset.
    If you put None in diameter, the model will estimate it automatically.
    Mask_threshold indicates how many of the possible masks are kept. Making it smaller (up to -6), will give you more masks, bigger is less masks.
    When an RGB image is given an input, the R channel is expected to have the nuclei, and the blue channel the membranes.
    When whole cell segmentation needs to be performed, model_type=cyto, otherwise, model_type=nuclei.
    """

    channels = np.array(channels)
    # Perform cellpose segmentation
    model = models.Cellpose(device=torch.device(device), model_type=model_type)
    masks, _, _, _ = model.eval(
        img.data.image.squeeze().to_numpy(),
        diameter=diameter,
        channels=channels,
        min_size=min_size,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
    )
    #mask_i = np.ma.masked_where(masks == 0, masks)

    # Create the polygon shapes of the different cells
    polygons = mask_to_polygons_layer(masks)
    #polygons["border_color"] = polygons.geometry.map(border_color)
    polygons["linewidth"] = polygons.geometry.map(linewidth)
    #polygons["color"] = polygons.geometry.map(color)
    polygons["cells"] = polygons.index
    polygons = polygons.dissolve(by="cells")
    polygons.index = list(map(str, polygons.index))

    #img.add_img(masks, layer="segment_cellpose")
    
    return masks, polygons, img


def segmentationPlot(
    sdata,
    crd=None,
    channels=[0,0],
    small_size_vis=None,
    img_layer='corrected',
    shape_layer=None,
    output: str = None,
) -> None:
    if output:
        plt.ioff()
    if small_size_vis:
        crd=small_size_vis
    ic=sdata.images[img_layer] 
    if shape_layer==None:
        shape_layer=[*sdata.shapes][0]
    Xmax=ic.sizes['x']
    Ymax=ic.sizes['y']    
    #ic=ic.data.assign_coords({'x':np.arange(Xmax),'y':np.arange(Ymax)})  
    #crd=small_size_vis
    #crd=[2000,3500,1000,2500]
    if crd==None:
        crd=[0,Ymax,0,Xmax]
    fig, ax = plt.subplots(1, 2, figsize=(20, 20))
    ic.squeeze().sel(x=slice(crd[2],crd[3]),y=slice(crd[0],crd[1])).plot.imshow(cmap='gray', robust=True, ax=ax[0],add_colorbar=False)
    ic.squeeze().sel(x=slice(crd[2],crd[3]),y=slice(crd[0],crd[1])).plot.imshow(cmap='gray', robust=True, ax=ax[1],add_colorbar=False)
    sdata[shape_layer].cx[crd[2]:crd[3],crd[0]:crd[1]].plot(
            ax=ax[1],
            edgecolor="white",
            linewidth=1,
            alpha=0.5,
            legend=True,
            aspect=1,
            )
    for i in range(len(ax)):
        ax[i].axes.set_aspect('equal')
        ax[i].set_xlim(crd[2], crd[3])
        ax[i].set_ylim(crd[0], crd[1])
        ax[i].invert_yaxis()
        ax[i].set_title("")
        #ax.axes.xaxis.set_visible(False)
        #ax.axes.yaxis.set_visible(False)
        ax[i].spines["top"].set_visible(False)
        ax[i].spines["right"].set_visible(False)
        ax[i].spines["bottom"].set_visible(False)
        ax[i].spines["left"].set_visible(False)
    
    # Save the plot to ouput
    if output:
        plt.close(fig)
        fig.savefig(output + ".png")            
        

def mask_to_polygons_layer(mask: np.ndarray) -> geopandas.GeoDataFrame:
    """Returns the polygons as GeoDataFrame

    This function converts the mask to polygons.
    https://rocreguant.com/convert-a-mask-into-a-polygon-for-images-using-shapely-and-rasterio/1786/
    """

    all_polygons = []
    all_values = []

    # Extract the polygons from the mask
    for shape, value in features.shapes(
        mask.astype(np.int32),
        mask=(mask > 0),
        transform=rasterio.Affine(1.0, 0, 0, 0, 1.0, 0),
    ):
        all_polygons.append(shapely.geometry.shape(shape))
        all_values.append(int(value))

    return geopandas.GeoDataFrame(dict(geometry=all_polygons), index=all_values)


def color(_) -> matplotlib.colors.Colormap:
    """Select random color from set1 colors."""
    return plt.get_cmap("Set1")(np.random.choice(np.arange(0, 18)))


def border_color(r: bool) -> matplotlib.colors.Colormap:
    """Select border color from tab10 colors or preset color (1, 1, 1, 1) otherwise."""
    return plt.get_cmap("tab10")(3) if r else (1, 1, 1, 1)


def linewidth(r: bool) -> float:
    """Select linewidth 1 if true else 0.5."""
    return 1 if r else 0.5

def delete_overlap(voronoi,polygons):    
    I1,I2=voronoi.sindex.query_bulk(voronoi['geometry'], predicate="overlaps")  
    voronoi2=voronoi.copy()

    for cell1, cell2 in zip(I1,I2):

        #if cell1!=cell2:
        voronoi.geometry.iloc[cell1]=voronoi.iloc[cell1].geometry.intersection(voronoi2.iloc[cell1].geometry.difference(voronoi2.iloc[cell2].geometry))
        voronoi.geometry.iloc[cell2]=voronoi.iloc[cell2].geometry.intersection(voronoi2.iloc[cell2].geometry.difference(voronoi2.iloc[cell1].geometry))
    voronoi['geometry']=voronoi.geometry.union(polygons.geometry) 
    polygons=polygons.buffer(distance=0) 
    voronoi=voronoi.buffer(distance=0)        
    return voronoi

def allocation(sdata, masks=None,radius=0,shape_layer=None, verbose=False
    ):
    
    # Create the polygon shapes for the mask
    if shape_layer==None:
        shape_layer=[*sdata.shapes][0]
    sdata[shape_layer].index=sdata[shape_layer].index.astype('str')
   
    #calculate new mask based on radius    
    if radius!=0:
        
        expanded_layer_name='expanded_cells'+str(radius)
        #sdata[shape_layer].index = list(map(str, sdata[shape_layer].index))
        
        boundary=Polygon([(0,0),(sdata[[*sdata.images][0]].shape[1]+200,0),(sdata[[*sdata.images][0]].shape[1]+200,sdata[[*sdata.images][0]].shape[2]+200),(0,sdata[[*sdata.images][0]].shape[2]+200)])
        if expanded_layer_name in [*sdata.shapes]:
            del sdata.shapes[expanded_layer_name]
        sdata[expanded_layer_name]=sdata[shape_layer].copy()
        sdata[expanded_layer_name]['geometry']=sdata[shape_layer].simplify(2)

        vd = voronoiDiagram4plg(sdata[expanded_layer_name], boundary)
        voronoi=geopandas.sjoin(vd,sdata[expanded_layer_name],predicate='contains',how='left')
        voronoi.index=voronoi.index_right
        voronoi=voronoi[~voronoi.index.duplicated(keep='first')]
        voronoi=delete_overlap(voronoi,sdata[expanded_layer_name])
        
        buffered=sdata[expanded_layer_name].buffer(distance=radius)
        intersected=voronoi.sort_index().intersection(buffered.sort_index())
        sdata[expanded_layer_name].geometry=intersected
        
        masks=rasterio.features.rasterize( #it should be possible to give every shape  number. You need to give the value with it as input. 
                    zip(intersected.geometry,intersected.index.values.astype(float)), 
                    out_shape=[sdata[[*sdata.images][0]].shape[1],sdata[[*sdata.images][0]].shape[2]],dtype='uint32')  
        
    # adapt transcripts file  
    sdata.add_points(name='selected_transcripts',points=sdata['transcripts'][
    (sdata['transcripts']['y'] < sdata['segmentation_mask'].shape[0])
    & (sdata['transcripts']['y'] >=0)
        & (sdata['transcripts']['x'] >=0)   
    & (sdata['transcripts']['x'] < sdata['segmentation_mask'].shape[1])
    ],overwrite=True)

    if verbose:
        print('Started df calculation')

    df=sdata['selected_transcripts'].compute()
    
    if masks is None:
        masks=sdata['segmentation_mask'].squeeze().to_numpy()

    if verbose:
        print('df calculated')
    df["cells"] = masks[
        df['y'].values.astype(int),
        df['x'].values.astype(int),
    ]

    # Calculate the mean of the transcripts for every cell
    coordinates = df.groupby(["cells"]).mean().loc[:, ['x', 'y']]
    cell_counts = df.groupby(["cells", "gene"]).size().unstack(fill_value=0)
    if verbose: 

        print('finished groupby')
        # Create the anndata object
    adata = AnnData(cell_counts[cell_counts.index != 0],dtype='int64')
    coordinates.index = coordinates.index.map(str)

    adata.obsm["spatial"] = coordinates[coordinates.index != "0"].values
    adata.obs['region']=1
    adata.obs['instance']=1

    if sdata.table:
        del sdata.table

    sdata.table=spatialdata.models.TableModel.parse(adata,region_key='region',region=1,instance_key='instance')


    for i in [*sdata.shapes]:
        sdata[i].index=sdata[i].index.astype('str')
        sdata.add_shapes(name=i,
        shapes=spatialdata.models.ShapesModel.parse(sdata[i][np.isin(sdata[i].index.values,sdata.table.obs.index.values)]),overwrite=True)
        #adata.uns["spatial"][library_id]["segmentation"] = masks.astype(np.uint16)

    return sdata,df


    for i in [*sdata.shapes]:
        sdata[i].index=sdata[i].index.astype('str')
        sdata.add_shapes(name=i,
        shapes=spatialdata.models.ShapesModel.parse(sdata[i][np.isin(sdata[i].index.values,sdata.table.obs.index.values)]),overwrite=True)
        #adata.uns["spatial"][library_id]["segmentation"] = masks.astype(np.uint16)

    return sdata,df


def control_transcripts(df,scaling_factor=100):
    """This function plots the transcript density of the tissue. You can use it to compare different regions in your tissue on transcript density.  """
    Try=df.groupby(['x','y']).count()['gene']
    Image=np.array(Try.unstack(fill_value=0))
    Image=Image/np.max(Image)
    blurred = gaussian_filter(scaling_factor*Image, sigma=7)
    return blurred



def plot_control_transcripts(blurred,sdata,crd=None):
    if crd:
        fig, ax = plt.subplots(1, 2, figsize=(20, 20))

        ax[0].imshow(blurred.T[crd[0]:crd[1],crd[2]:crd[3]],cmap='magma',vmax=5)
        sdata['corrected'].squeeze().sel(x=slice(crd[0],crd[1]),y=slice(crd[2],crd[3])).plot.imshow(cmap='gray', robust=True, ax=ax[1],add_colorbar=False)
        ax[0].set_title("Transcript density")
        ax[1].set_title("Corrected image")
    fig, ax = plt.subplots(1, 2, figsize=(20, 20))

    ax[0].imshow(blurred.T,cmap='magma',vmax=5)
    sdata['corrected'].squeeze().plot.imshow(cmap='gray', robust=True, ax=ax[1],add_colorbar=False)
    ax[1].axes.set_aspect('equal')
    ax[1].invert_yaxis()

    ax[0].set_title("Transcript density")
    ax[1].set_title("Corrected image")
    
    

def analyse_genes_left_out(sdata,df):
    """ This function """
    filtered=pd.DataFrame(sdata.table.X.sum(axis=0)/df.groupby('gene').count()['x'][sdata.table.var.index])
    filtered=filtered.rename(columns={'x':'proportion_kept'})
    filtered['raw_counts']=df.groupby('gene').count()['x'][sdata.table.var.index]
    filtered['log_raw_counts']=np.log(filtered['raw_counts'])

    sns.scatterplot(data=filtered, y="proportion_kept", x="log_raw_counts")
    
    plt.axvline(filtered['log_raw_counts'].median(), color='green', linestyle='dashed')
    plt.axhline(filtered['proportion_kept'].median(), color='red', linestyle='dashed')
    plt.xlim(left=-0.5,right=filtered['log_raw_counts'].quantile(0.99)) # set y-axis limit from 0 to the 95th percentile of y
    # show the plot
    plt.show()
    r, p = sp.stats.pearsonr(filtered['log_raw_counts'], filtered['proportion_kept'])
    sns.regplot(x = "log_raw_counts", y = "proportion_kept", data = filtered)
    ax=plt.gca()
    ax.text(0.7, 0.9, 'r={:.2f}, p={:.2g}'.format(r, p),
            transform=ax.transAxes)
    
    plt.axvline(filtered['log_raw_counts'].median(), color='green', linestyle='dashed')
    plt.axhline(filtered['proportion_kept'].median(), color='red', linestyle='dashed')
    plt.show()
    print('The ten genes with the highest proportion of transcripts filtered out')
    print(filtered.sort_values(by='proportion_kept')[0:10].iloc[:,0:2])
    return filtered


def create_adata_quick(
    path: str, ic: sq.im.ImageContainer, masks: np.ndarray, library_id: str = "spatial_transcriptomics"
) -> AnnData:
    """Returns the AnnData object with transcript and polygon data.

    This function creates the polygon shapes from the mask and adjusts the colors and linewidth.
    The transcripts are read from the csv file in path, all transcripts within cells are assigned.
    Only cells with transcripts are retained.
    """

    # Create the polygon shapes of the different cells
    polygons = mask_to_polygons_layer(masks)
    polygons["geometry"] = polygons["geometry"].translate(
        float(ic.data.attrs["coords"].x0), float(ic.data.attrs["coords"].y0)
    )

    polygons["border_color"] = polygons.geometry.map(border_color)
    polygons["linewidth"] = polygons.geometry.map(linewidth)
    polygons["color"] = polygons.geometry.map(color)
    polygons["cells"] = polygons.index
    polygons = polygons.dissolve(by="cells")

    # Allocate the transcripts
    in_df = pd.read_csv(path, delimiter="\t", header=None)
    # Changed for subset
    df = in_df[
        (ic.data.attrs["coords"].y0 < in_df[1])
        & (in_df[1] < masks.shape[0] + ic.data.attrs["coords"].y0)
        & (ic.data.attrs["coords"].x0 < in_df[0])
        & (in_df[0] < masks.shape[1] + ic.data.attrs["coords"].x0)
    ]

    df["cells"] = masks[
        df[1].values - ic.data.attrs["coords"].y0,
        df[0].values - ic.data.attrs["coords"].x0,
    ]

    # Calculate the mean of the transcripts for every cell
    coordinates = df.groupby(["cells"]).mean().iloc[:, [0, 1]]
    cell_counts = df.groupby(["cells", 3]).size().unstack(fill_value=0)

    # Create the anndata object
    adata = AnnData(cell_counts[cell_counts.index != 0])
    coordinates.index = coordinates.index.map(str)
    adata.obsm["spatial"] = coordinates[coordinates.index != "0"]#.values

    # Add the polygons to the anndata object
    polygons_f = polygons[
        np.isin(polygons.index.values, list(map(int, adata.obs.index.values)))
    ]
    polygons_f.index = list(map(str, polygons_f.index))
    adata.obsm["polygons"] = polygons_f
    adata.obs["cell_ID"] = [int(x) for x in adata.obsm["polygons"].index]

    # Add the figure to the anndata object
    adata.uns["spatial"] = {library_id: {}}
    adata.uns["spatial"][library_id]["images"] = {}
    adata.uns["spatial"][library_id]["images"] = {
        "hires": ic.data.image.squeeze().to_numpy()
    }
    adata.uns["spatial"][library_id]["scalefactors"] = {
        "tissue_hires_scalef": 1,
        "spot_diameter_fullres": 75,
    }
    adata.uns["spatial"][library_id]["segmentation"] = masks.astype(np.uint16)
    adata.uns["spatial"][library_id]["points"] = AnnData(in_df.values[:, 0:2])
    adata.uns["spatial"][library_id]["points"].obs = pd.DataFrame(
        {"gene": in_df.values[:, 3]}
    )

    return adata


def plot_shapes(
    sdata,
    column: str = None,
    cmap: str = "magma",
    img_layer='corrected',
    shapes_layer=None,
    alpha: float = 0.5,
    library_id='spatial_transcriptomics',
    crd=None,
    output: str = None,
    vmin=None,
    vmax=None,
) -> None:
    
    
    
    
    Xmax=sdata[img_layer].sizes['x']
    Ymax=sdata[img_layer].sizes['y']    
    
    if crd==None:
        crd=[0,Xmax,0,Ymax]
    if shapes_layer==None:
        shapes_layer=[*sdata.shapes][0]
        
    if column is not None:
        if column + "_colors" in sdata.table.uns:
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                "new_map",
                sdata.table.uns[column + "_colors"],
                N=len(sdata.table.uns[column + "_colors"]),
            ) 
        if column in sdata.table.obs.columns:    
            column=sdata.table[sdata[shapes_layer].cx[crd[0]:crd[1],crd[2]:crd[3]].index,:].obs[column]
        elif column in sdata.table.var.index:
            column=sdata.table[sdata[shapes_layer].cx[crd[0]:crd[1],crd[2]:crd[3]].index,:].X[:,np.where(sdata.table.var.index==column)[0][0]]
        else: 
            print('The column defined in the function isnt a column in obs, nor is it a gene name, the plot is made without taking into account this value.')
            column= None
            cmap=None
    else:
        cmap=None
    if vmin!=None:
        vmin=np.percentile(column,vmin)
    if vmax!=None:
        vmax=np.percentile(column,vmax)    
    
    fig, ax = plt.subplots(1, 1, figsize=(20, 20))

    sdata[img_layer].squeeze().sel(x=slice(crd[0],crd[1]),y=slice(crd[2],crd[3])).plot.imshow(cmap='gray', robust=True, ax=ax,add_colorbar=False)

    sdata[shapes_layer].cx[crd[0]:crd[1],crd[2]:crd[3]].plot(
            ax=ax,
            edgecolor="white",
            column=column,
            linewidth=1,
            alpha=alpha,
            legend=True,
            aspect=1,
            cmap=cmap,
            vmax=vmax,#np.percentile(column,vmax),
            vmin=vmin,#np.percentile(column,vmin)
            )
    
    ax.axes.set_aspect('equal')
    ax.set_xlim(crd[0], crd[1])
    ax.set_ylim(crd[2], crd[3])
    ax.invert_yaxis()
    ax.set_title("")
    #ax.axes.xaxis.set_visible(False)
    #ax.axes.yaxis.set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Save the plot to ouput
    if output:
        plt.close(fig)
        fig.savefig(output + ".png")
    else:
        plt.show()

def preprocessAdata(
    sdata, nuc_size_norm: bool = True, n_comps: int = 50,min_counts=10,min_cells=5,shapes_layer=None):
    """Returns the new and original AnnData objects

    This function calculates the QC metrics.
    All cells with les then 10 genes and all genes with less then 5 cells are removed.
    Normalization is performed based on the size of the nucleus in nuc_size_norm."""
    # calculate the max amount of pc's possible 
    if min(sdata.table.shape)<n_comps:
        
        n_comps=min(sdata.table.shape)
        print('amount of pc\'s was set to '+str(min(sdata.table.shape)),' because of the dimensionality of the data.')
    # Calculate QC Metrics
    
    sc.pp.calculate_qc_metrics(sdata.table, inplace=True, percent_top=[2, 5])

    # Filter cells and genes
    sc.pp.filter_cells(sdata.table, min_counts=min_counts)
    sc.pp.filter_genes(sdata.table, min_cells=min_cells)
    sdata.table.raw= sdata.table

    # Normalize nucleus size
    if shapes_layer is None:
        shapes_layer=[*sdata.shapes][-1]
    sdata.table.obs["shapeSize"] = sdata[shapes_layer].area
    
    if nuc_size_norm:   
        sdata.table.X = (sdata.table.X.T*100 / sdata.table.obs.shapeSize.values).T

        sc.pp.log1p(sdata.table)
        sc.pp.scale(sdata.table, max_value=10)
    else:
        sc.pp.normalize_total(sdata.table)
        sc.pp.log1p(sdata.table)

    sc.tl.pca(sdata.table, svd_solver="arpack", n_comps=n_comps)
    #Is this the best way o doing it? Every time you subset your data, the polygons should be subsetted too! 
    
    for i in [*sdata.shapes]:
        sdata[i].index=sdata[i].index.astype('str')
        sdata.add_shapes(name=i,
        shapes=spatialdata.models.ShapesModel.parse(sdata[i][np.isin(sdata[i].index.values,sdata.table.obs.index.values)]),overwrite=True)

    return sdata


def preprocesAdataPlot(sdata, output: str = None) -> None:
    """This function plots the size of the nucleus related to the counts."""

    # disable interactive mode
    if output:
        plt.ioff()

    sc.pl.pca(sdata.table, color="total_counts", show=not output,title='PC plot colored by total counts')
    sc.pl.pca(sdata.table, color="shapeSize", show=not output,title='PC plot colored by object size')

    fig, axs = plt.subplots(1, 2, figsize=(15, 4))
    sns.histplot(sdata.table.obs["total_counts"], kde=False, ax=axs[0])
    sns.histplot(sdata.table.obs["n_genes_by_counts"], kde=False, bins=55, ax=axs[1])
    plt.show()
    plt.scatter(sdata.table.obs["shapeSize"], sdata.table.obs["total_counts"])
    plt.title = "cellsize vs cellcount"

    # Save the plot to ouput
    if output:
        plt.close()
        plt.savefig(output + "0.png")
        plt.close(fig)
        fig.savefig(output + "1.png")


def filter_on_size(
    sdata, min_size: int = 100, max_size: int = 100000
):
    """Returns a tuple with the AnnData object and the number of filtered cells.

    All cells outside of the min and max size range are removed.
    If the distance between the location of the transcript and the center of the polygon is large, the cell is deleted.
    """

    start = sdata.table.shape[0]

    # Filter cells based on size and distance
    table = sdata.table[sdata.table.obs["shapeSize"] < max_size, :]
    table = table[table.obs["shapeSize"] > min_size, :]
    del sdata.table 
    ## TODO: Look for a better way of doing this!
    sdata.table=spatialdata.models.TableModel.parse(table)
    
    for i in [*sdata.shapes]:
        sdata[i].index=sdata[i].index.astype('str')
        sdata.add_shapes(name=i,
        shapes=spatialdata.models.ShapesModel.parse(sdata[i][np.isin(sdata[i].index.values,sdata.table.obs.index.values)]),overwrite=True)
    filtered = start - table.shape[0]
    print(str(filtered) + " cells were filtered out based on size.")


    return sdata


##TODO:rewrite this function
def extract(ic: sq.im.ImageContainer, adata: AnnData) -> AnnData:
    """This function performs segmenation feature extraction and adds cell area and mean intensity to the annData object under obsm segmentation_features."""
    sq.im.calculate_image_features(
        adata,
        ic,
        layer="raw",
        features="segmentation",
        key_added="segmentation_features",
        features_kwargs={
            "segmentation": {
                "label_layer": "segment_cellpose",
                "props": ["label", "area", "mean_intensity"],
                "channels": [0],
            }
        },
    )

    return adata


def clustering(
    sdata, pcs: int, neighbors: int, cluster_resolution: float = 0.8
) -> AnnData:
    """Returns the AnnData object.

    Performs neighborhood analysis, Leiden clustering and UMAP.
    Provides option to save the plots to output.
    """

    # Neighborhood analysis
    sc.pp.neighbors(sdata.table, n_neighbors=neighbors, n_pcs=pcs)
    sc.tl.umap(sdata.table)

    # Leiden clustering
    sc.tl.leiden(sdata.table, resolution=cluster_resolution)
    sc.tl.rank_genes_groups(sdata.table, "leiden", method="wilcoxon")

    return sdata


def clustering_plot(sdata, output: str = None) -> None:
    """This function plots the clusters and genes ranking"""

    # disable interactive mode
    if output:
        plt.ioff()

    # Leiden clustering
    sc.pl.umap(sdata.table, color=["leiden"], show=not output)

    # Save the plot to ouput
    if output:
        plt.savefig(output + "0.png", bbox_inches="tight")
        plt.close()
        sc.pl.rank_genes_groups(sdata.table, n_genes=8, sharey=False, show=False)
        plt.savefig(output + "1.png", bbox_inches="tight")
        plt.close()

    # Display plot
    else:
        sc.pl.rank_genes_groups(sdata.table, n_genes=8, sharey=False)


def scoreGenes(
    sdata,
    path_marker_genes: str,
    row_norm: bool = False,
    repl_columns: dict[str, str] = None,
    del_celltypes: List[str] = None,
    input_dict=False
) -> Tuple[dict, pd.DataFrame]:
    """Returns genes dict and the scores per cluster

    Load the marker genes from csv file in path_marker_genes.
    If the marker gene list is a one hot endoded matrix, leave the input as is. 
    IF the marker gene list is a Dictionary, with the first column the name of the celltype and the other columns the marker genes beloning to this celltype, 
    input 
    repl_columns holds the column names that should be replaced the in the marker genes.
    del_genes holds the marker genes that should be deleted from the marker genes and genes dict.
    """

    # Load marker genes from csv
    if input_dict:
        df_markers=pd.read_csv(path_marker_genes,header=None,index_col=0)
        df_markers=df_markers.T
        genes_dict=df_markers.to_dict('list')
        for i in genes_dict:
            genes_dict[i]=[x for x in genes_dict[i] if str(x)!='nan']
    # Replace column names in marker genes
    else:
        df_markers = pd.read_csv(path_marker_genes, index_col=0)
        if repl_columns:
            for column, replace in repl_columns.items():
                df_markers.columns = df_markers.columns.str.replace(column, replace)

        # Create genes dict with all marker genes for every celltype
        genes_dict = {}
        for i in df_markers:
            genes = []
            for row, value in enumerate(df_markers[i]):
                if value > 0:
                    genes.append(df_markers.index[row])
            genes_dict[i] = genes

    # Score all cells for all celltypes
    for key, value in genes_dict.items():
        try:
            sc.tl.score_genes(sdata.table, value, score_name=key)
        except ValueError:
            warnings.warn(
                f"Markergenes {value} not present in region, celltype {key} not found"
            )

    # Delete genes from marker genes and genes dict
    if del_celltypes:
        for gene in del_celltypes:
            del df_markers[gene]
            del genes_dict[gene]

    scoresper_cluster = sdata.table.obs[
        [col for col in sdata.table.obs if col in df_markers.columns]
    ]

    # Row normalization for visualisation purposes
    if row_norm:
        row_norm = scoresper_cluster.sub(
            scoresper_cluster.mean(axis=1).values, axis="rows"
        ).div(scoresper_cluster.std(axis=1).values, axis="rows")
        sdata.table.obs[scoresper_cluster.columns.values] = row_norm
        temp = pd.DataFrame(np.sort(row_norm)[:, -2:])
    else:
        temp = pd.DataFrame(np.sort(scoresper_cluster)[:, -2:])

    scores = (temp[1] - temp[0]) / ((temp[1] + temp[0]) / 2)
    sdata.table.obs["Cleanliness"] = scores.values
    sdata.table.obs["annotation"] = scoresper_cluster.idxmax(axis=1)
    sdata.table.obs["annotation"] = sdata.table.obs["annotation"].astype('category')
    return genes_dict, scoresper_cluster


def scoreGenesPlot(
    sdata,
    scoresper_cluster: pd.DataFrame,
    filter_index: int = 5,
    output: str = None,
    library_id='spatial_transcriptomics'
) -> None:
    """This function plots the cleanliness and the leiden score next to the annotation."""

    # Custom colormap:
    colors = np.concatenate(
        (plt.get_cmap("tab20c")(np.arange(20)), plt.get_cmap("tab20b")(np.arange(20)))
    )
    colors = [mpl.rgb2hex(colors[j * 4 + i]) for i in range(4) for j in range(10)]

    # disable interactive mode
    if output:
        plt.ioff()

    # Plot cleanliness and leiden next to annotation
    sc.pl.umap(sdata.table, color=["Cleanliness", "annotation"], show=not output)

    # Save the plot to ouput
    if output:
        plt.savefig(output + "0.png", bbox_inches="tight")
        plt.close()
        sc.pl.umap(sdata.table, color=["leiden", "annotation"], show=False)
        plt.savefig(output + "1.png", bbox_inches="tight")
        plt.close()

    # Display plot
    else:
        sc.pl.umap(sdata.table, color=["leiden", "annotation"])

    # Plot annotation and cleanliness columns of AnnData object
    sdata.table.uns["annotation_colors"] = colors
    plot_shapes(sdata, column="annotation", output=output + "2" if output else None)
    #plot_shapes(sdata.table, column="Cleanliness", output=output + "3" if output else None,library_id=library_id)

    # Plot heatmap of celltypes and filtered celltypes based on filter index
    sc.pl.heatmap(
        sdata.table,
        var_names=scoresper_cluster.columns.values,
        groupby="leiden",
        show=not output,
    )

    # Save the plot to ouput
    if output:
        plt.savefig(output + "4.png", bbox_inches="tight")
        plt.close()
        sc.pl.heatmap(
            sdata.table[
                sdata.table.obs.leiden.isin(
                    [str(index) for index in range(filter_index, len(sdata.table.obs.leiden))]
                )
            ],
            var_names=scoresper_cluster.columns.values,
            groupby="leiden",
            show=False,
        )
        plt.savefig(output + "5.png", bbox_inches="tight")
        plt.close()

    # Display plot
    else:
        sc.pl.heatmap(
            sdata.table[
                sdata.table.obs.leiden.isin(
                    [str(index) for index in range(filter_index, len(sdata.table.obs.leiden))]
                )
            ],
            var_names=scoresper_cluster.columns.values,
            groupby="leiden",
        )

def correct_marker_genes(
    sdata, Dict
):
    """Returns the new AnnData object.

    Corrects marker genes that are higher expessed by dividing them.
    The genes has as keys the genes that should be corrected and as values the threshold and the divider.
    """

    # Correct for all the genes
    for gene, values in Dict.items():
        sdata.table.obs[gene]=np.where(sdata.table.obs[gene]<values[0],sdata.table.obs[gene]/values[1],sdata.table.obs[gene])
    
    return sdata

def annotate_maxscore(types: str, indexes: dict, sdata):
    """Returns the AnnData object.

    Adds types to the Anndata maxscore category.
    """
    sdata.table.obs.annotation = sdata.table.obs.annotation.cat.add_categories([types])
    for i, val in enumerate(sdata.table.obs.annotation):
        if val in indexes[types]:
            sdata.table.obs.annotation[i] = types
    return sdata


def remove_celltypes(types: str, indexes: dict, sdata):
    """Returns the AnnData object."""
    for index in indexes[types]:
        if index in sdata.table.obs.annotation.cat.categories:
            sdata.table.obs.annotation = sdata.table.obs.annotation.cat.remove_categories(index)
    return sdata


def clustercleanliness(
    sdata,
    genes: List[str],
    gene_indexes: dict[str, int] = None,
    colors: List[str] = None,
) -> Tuple[AnnData, Optional[dict]]:
    """Returns a tuple with the AnnData object and the color dict."""
    celltypes = np.array(sorted(genes), dtype=str)
    color_dict = None

    sdata.table.obs["annotation"] = sdata.table.obs[
        [col for col in sdata.table.obs if col in celltypes]
    ].idxmax(axis=1)
    sdata.table.obs.annotation = sdata.table.obs.annotation.astype("category")

    # Create custom colormap for clusters
    if not colors:
        color = np.concatenate(
            (
                plt.get_cmap("tab20c")(np.arange(20)),
                plt.get_cmap("tab20b")(np.arange(20)),
            )
        )
        colors = [mpl.rgb2hex(color[j * 4 + i]) for i in range(4) for j in range(10)]

    sdata.table.uns["annotation_colors"] = colors

    if gene_indexes:
        sdata.table.obs["annotationSave"] = sdata.table.obs.annotation
        gene_celltypes = {}

        for key, value in gene_indexes.items():
            gene_celltypes[key] = celltypes[value]

        for gene, indexes in gene_indexes.items():
            sdata = annotate_maxscore(gene, gene_celltypes, sdata)

        for gene, indexes in gene_indexes.items():
            sdata = remove_celltypes(gene, gene_celltypes, sdata)

        celltypes_f = np.delete(celltypes, list(chain(*gene_indexes.values())))  # type: ignore
        celltypes_f = np.append(celltypes_f, list(gene_indexes.keys()))
        color_dict = dict(zip(celltypes_f, sdata.table.uns["annotation_colors"]))

    else:
        color_dict = dict(zip(celltypes, sdata.table.uns["annotation_colors"]))

    for i, name in enumerate(color_dict.keys()):
        color_dict[name] = colors[i]
    sdata.table.uns["annotation_colors"] = list(
        map(color_dict.get, sdata.table.obs.annotation.cat.categories.values)
    )

    return sdata, color_dict


def clustercleanlinessPlot(
    sdata,
    crop_coord: List[int] = [0, 2000, 0, 2000],
    color_dict: dict = None,
    output: str = None,
    library_id='spatial_transcriptomics'
) -> None:
    """This function plots the clustercleanliness as barplots, the images with colored celltypes and the clusters."""

    # disable interactive mode
    if output:
        plt.ioff()

    # Create the barplot
    stacked = (
        sdata.table.obs.groupby(["leiden", "annotation"], as_index=False)
        .size()
        .pivot("leiden", "annotation")
        .fillna(0)
    )
    stacked_norm = stacked.div(stacked.sum(axis=1), axis=0)
    stacked_norm.columns = list(sdata.table.obs.annotation.cat.categories)
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    # Use custom colormap
    if color_dict:
        stacked_norm.plot(kind="bar", stacked=True, ax=fig.gca(), color=color_dict)
    else:
        stacked_norm.plot(kind="bar", stacked=True, ax=fig.gca())
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.get_yaxis().set_ticks([])
    plt.xlabel("Clusters")
    plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize="large")

    # Save the barplot to ouput
    if output:
        plt.close(fig)
        fig.savefig(output + "0.png", bbox_inches="tight")
    else:
        plt.show()

    # Plot images with colored celltypes
    plot_shapes(
        sdata, column="annotation", alpha=0.8, output=output + "1" if output else None, library_id=library_id
    )
    plot_shapes(
        sdata,
        column="annotation",
        crd=crop_coord,
        alpha=0.8,
        output=output + "2" if output else None,
        library_id=library_id
    )

    # Plot clusters
    _, ax = plt.subplots(1, 1, figsize=(15, 10))
    sc.pl.umap(sdata.table, color=["annotation"], ax=ax, show=not output,size=300000/sdata.table.shape[0])
    ax.axis("off")

    # Save the plot to ouput
    if output:
        fig.savefig(output + "3.png", bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def enrichment(sdata):
    """Returns the AnnData object.

    Performs some adaptations to save the data.
    Calculate the nhood enrichment"
    """

    # Adaptations for saving
    sdata.table.raw.var.index.names = ["genes"]
    sdata.table.var.index.names = ["genes"]
    # TODO: not used since napari spatialdata
    # adata.obsm["spatial"] = adata.obsm["spatial"].rename({0: "X", 1: "Y"}, axis=1)

    # Calculate nhood enrichment
    sq.gr.spatial_neighbors(sdata.table, coord_type="generic")
    sq.gr.nhood_enrichment(sdata.table, cluster_key="annotation")
    return sdata


def enrichment_plot(sdata, output: str = None) -> None:
    """This function plots the nhood enrichment between different celltypes."""

    # disable interactive mode
    if output:
        plt.ioff()
    # remove 'nan' values from "adata.uns['annotation_nhood_enrichment']['zscore']"
    tmp = sdata.table.uns['annotation_nhood_enrichment']['zscore']
    sdata.table.uns['annotation_nhood_enrichment']['zscore'] = np.nan_to_num(tmp)
    sq.pl.nhood_enrichment(sdata.table, cluster_key="annotation", method="ward")

    # Save the plot to ouput
    if output:
        plt.savefig(output + ".png", bbox_inches="tight")


def save_data(adata: AnnData, output_geojson: str, output_h5ad: str):
    """Saves the ploygons to output_geojson as GeoJson object and the rest of the AnnData object to output_h5ad as h5ad file."""

    # Save polygons to geojson
    if color in adata.obsm["polygons"]:
        del adata.obsm["polygons"]["color"]
    adata.obsm["polygons"]["geometry"].to_file(output_geojson, driver="GeoJSON")
    adata.obsm["polygons"] = pd.DataFrame(
        {
            "linewidth": adata.obsm["polygons"]["linewidth"],
             #"X": adata.obsm["polygons"]["X"],
            #"Y": adata.obsm["polygons"]["Y"],
        }
    )

    # Write AnnData object to h5ad file
    adata.write(output_h5ad)

def micron_to_pixels(df, offset_x=45_000,offset_y=45_000,pixelSize=None):
    if pixelSize:
        df[x] /= pixelSize
        df[y] /= pixelSize
    if offset_x:
            df['x'] -= offset_x
    if offset_y:        
            df['y'] -= offset_y

    return df
    
def read_in_stereoSeq(path_genes,xcol='x',ycol='y',genecol='geneID',countcol='MIDCount',skiprows=0,offset=None):
    """This function read in Stereoseq input data to a dask datafrmae with predefined column names. 
    As we are working with BGI data, a column with counts is added."""
    in_df=dd.read_csv(path_genes,delimiter='\t',skiprows=skiprows)
    in_df=in_df.rename(columns={xcol:'x',ycol:'y',genecol:'gene',countcol:'counts'})
    if offset:
        in_df=micron_to_pixels(in_df,offset_x=offset[0],offset_y=offset[1])
    in_df=in_df.loc[:,['x','y','gene','counts']]
    
    in_df=in_df.dropna()
    return in_df

def read_in_RESOLVE(path_coordinates,sdata=None,xcol=0,ycol=1,genecol=3,filterGenes=None,offset=None):
    
    """The output of this function gives all locations of interesting transcripts in pixel coordinates matching the input image. Dask Dataframe contains columns x,y, and gene"""
    
    if sdata==None:
        sdata=spatialdata.SpatialData()
    in_df = dd.read_csv(path_coordinates, delimiter="\t", header=None)
    in_df=in_df.rename(columns={xcol:'x',ycol:'y',genecol:'gene'})
    if offset:
        in_df=micron_to_pixels(in_df,offset_x=offset[0],offset_y=offset[1])
    in_df=in_df.loc[:,['x','y','gene']]
    in_df=in_df.dropna()
    
    if filterGenes:
        for i in filter_genes:
            in_df=in_df[in_df['gene'].str.contains(i)==False]
    sdata.add_points(name='transcripts',points=spatialdata.models.PointsModel.parse(in_df,coordinates={'x':'x','y':'y'}))
        
            
    return sdata

def read_in_Vizgen(path_genes,sdata=None,xcol='global_x',ycol='global_y',genecol='gene',skiprows=None,offset=None, bbox=None,pixelSize=None,filterGenes=None):
    """This function read in Vizgen input data to a dask datafrmae with predefined column names. """
    if sdata==None:
        sdata=spatialdata.SpatialData()
    in_df=dd.read_csv(path_genes,skiprows=skiprows,usecols=[xcol,ycol,genecol]) #speedu
    #in_df=in_df.loc[:,[xcol,ycol,genecol]]
    in_df=in_df.rename(columns={xcol:'x',ycol:'y',genecol:'gene'})
    if bbox:
        in_df['x']-=bbox[0]
        in_df['y']-=bbox[1]
    if pixelSize:
        in_df['x'] /= pixelSize
        in_df['y'] /= pixelSize
    if offset:
            in_df['x'] -= offset[0]   
            in_df['y'] -= offset[1]
    
    in_df=in_df.dropna()
    
    if filterGenes:
        for i in filterGenes:
            in_df=in_df[in_df['gene'].str.contains(i)==False]
    sdata.add_points(name='transcripts',points=spatialdata.models.PointsModel.parse(in_df,coordinates={'x':'x','y':'y'}))        
    return sdata


    

# %%
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 8:41:34 2019

Hace WB y parchea. Hay que lanzar en train,test

@author: Fernando Perez
"""

import xml.etree.ElementTree as ET
import math 
from PIL import Image, ImageDraw 
from PIL import ImagePath  
import openslide as sld

import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from skimage import filters
from skimage import exposure
from scipy import ndimage
#import cv2

import csv

camelyon_dir='/work/Camelyon16/'
annotation_dir='annotation_train'
annotation_mask_dir='annotation_masks'
save_dir='224'
image_dir='normal'
folder='training/'

#Para obtener dataset balanceado
cancer_patch_WSI=500
benign_patch_WSI=500

if not os.path.isdir(camelyon_dir+folder+save_dir):
    #os.mkdir(camelyon_dir+folder+save_dir)
    print('save dir created')
    
    
##### CUIDADO ABRIR IMAGENES MUY GRANDES PUEDE ROMPERLO TODO ########
# Elimina el limite de seguidad para poder abrir imagenes grandes.
Image.MAX_IMAGE_PIXELS = None
#####################################################################

# %%
#Para cada imagen WSI
for img_file in glob.glob(camelyon_dir+folder+image_dir+'/*.tif')[142:143]:
    Nmi_I=0
    Nmi_I_wb=0
    print (img_file)
    file=img_file.split("/")[-1]
    name=file.split(".")[0]
    #Leemos WSI
    image_slide = sld.open_slide(camelyon_dir+folder+image_dir+'/'+name+'.tif')
    size=image_slide.dimensions
    
    #Leemos mascara de anotacion
    annotation_file=camelyon_dir+folder+annotation_mask_dir+'/'+name+'_annotation_mask.png'
    is_annotated=False
    if os.path.isfile(annotation_file):
        is_annotated= True
        annotation_mask = Image.open(camelyon_dir+folder+annotation_mask_dir+'/'+name+'_annotation_mask.png')
        annotation_array = np.asarray(annotation_mask)
        thumb_annotation=annotation_mask.copy()
        thumb_annotation.thumbnail((5000,5000))
        thumb_annotation=np.asarray(thumb_annotation)
        annotation_size=np.sum(thumb_annotation)

    
    #Extraemos un thumb para visualización y WB
    thumb=image_slide.get_thumbnail((5000,5000))
    #plt.imshow(np.asarray(thumb))
    if name=='normal_144':
        thumb_array=np.asarray(thumb)
        thumb_array.setflags(write=1)
        thumb_array[thumb_array==0]=255
        thumb=Image.fromarray(thumb_array)

    #Pasamos a gris para extraer mascara de fondo
    #Image to gray
    thumb_gray=thumb.convert('LA')
    thumb_gray_array=np.asarray(thumb_gray)
    val_thumb = filters.threshold_otsu(thumb_gray_array[:,:,0])
    tissue_thumb=thumb_gray_array[:,:,0]<val_thumb
    fondo_thumb=thumb_gray_array[:,:,0]>200
    #plt.imshow(tissue_thumb,cmap='gray')
    
    #Sacamos valores para WB
    thumb_array=np.asarray(thumb)#[~tissue_thumb,:]
    Fondo_thumb=thumb_array[fondo_thumb,:]
    Fondo_thumb.shape
    white_balance=[np.mean(Fondo_thumb[:,0]),np.mean(Fondo_thumb[:,1]),np.mean(Fondo_thumb[:,2])]
    print("white balance: "+ str(white_balance))
        
    
    patch_size=224
    overlap=0 #En pixeles
    tumor_overlap=0 #En pixeles
    level=0
    
    min_healthy_tissue=0.7
    min_tumor=0.7
    
    patch_folder=camelyon_dir+folder+image_dir+'_patches'+str(patch_size)
    if not os.path.isdir(patch_folder):
        os.mkdir(patch_folder)
        print(patch_folder, 'created')
    
    image_folder=patch_folder+'/'+name
    if not os.path.isdir(image_folder):
        os.mkdir(image_folder)
        os.mkdir(image_folder+'/no_annotated')
        if is_annotated:
            os.mkdir(image_folder+'/annotated')
        
        
    white_file=image_folder+'/white_balance.csv'
    with open(white_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(white_balance)
    
    #ROI ESTIMATION##################################################################
    #Comprobar que regiones tienen el tejido y la anotación para buscar parches de forma más eficiente

    n_side=10 #n particiones en cada eje
    x_roi_size=int(tissue_thumb.shape[0]/n_side)
    y_roi_size=int(tissue_thumb.shape[1]/n_side)
    tissue_size=np.sum(tissue_thumb)
    
    #Conversion de thumb a slide
    x_rate=size[0]/tissue_thumb.shape[1]
    y_rate=size[1]/tissue_thumb.shape[0]
    
    index=1

    for x in range(0,tissue_thumb.shape[0]-1,x_roi_size):
        for y in range(0,tissue_thumb.shape[1]-1,y_roi_size):
            #plt.figure(1)
            #plt.subplot(n_side,n_side,index)
        
            #plt.imshow(tissue_thumb[x:x+x_roi_size-1,y:y+y_roi_size-1])
            #plt.axis('off')
            
            roi_tissue_percent=np.sum(tissue_thumb[x:x+x_roi_size-1,y:y+y_roi_size-1])/tissue_size
            annotation_percent=0
            if is_annotated:
                annotation_percent=np.sum(thumb_annotation[x:x+x_roi_size-1,y:y+y_roi_size-1])/annotation_size
            #print('Checking ROI '+str(index)+' tissue_percent: '+str(roi_tissue_percent) + ' annotation' + str(annotation_percent))
            savedNA=0
            savedT=0
            if (roi_tissue_percent>0.01 and index not in range(0,11)) or annotation_percent!=0:
                print('Parcheando ROI '+str(index)+' tissue_percent: '+str(roi_tissue_percent) + ' annotation' + str(annotation_percent))
                if annotation_percent!=0:
                    print('cancer region setup')
                    step=patch_size-tumor_overlap
                    min_tissue=min_tumor
                else:
                    step=patch_size-overlap
                    min_tissue=min_healthy_tissue

                #plt.imshow(tissue_thumb[x:x+x_roi_size-1,y:y+y_roi_size-1])
                #plt.axis('off')
                #La ROI contiene tejido suficiente para analizarla. Buscamos parches de interes
                x_slide=int(x*x_rate)
                y_slide=int(y*y_rate)
                end_roi_x=int((x+x_roi_size)*x_rate)
                end_roi_y=int((y+y_roi_size)*y_rate)
                
                max_x=end_roi_x-patch_size+1
                max_y=end_roi_y-patch_size+1
                
                #Iteramos la region de arriba a abajo y de izq a derecha sacando parches
                for x0_patch in range(x_slide,max_x,step):
                    for y0_patch in range(y_slide,max_y,step):
                        #Para cada parche
                        upper_left_corner=(y0_patch,x0_patch)
                        patch=image_slide.read_region(upper_left_corner, level, (patch_size,patch_size))
                        patch_array=np.asarray(patch)
                        
                        #Parche a gris para mascara de tejido
                        patch_gray=np.asarray(patch.convert('LA'))
                        patch_tissue=patch_gray[:,:,0]<val_thumb
                        patch_tissue=ndimage.binary_dilation(patch_tissue,iterations=2)
                        patch_tissue_percent=np.sum(patch_tissue)/patch_tissue.size
                        
                        #Primer parche debugging
                        #plt.figure()
                        #plt.imshow(patch_array)
                        #print(patch_tissue_percent)
                        
                        #Si el parche contiene tejido suficiente
                        if patch_tissue_percent > min_tissue and savedNA<benign_patch_WSI:
                            
                            #print("saving patch")
                            #patch_annotation=np.asarray(annotation_mask)[y:y+patch_size,x:x+patch_size]
                            #tumor_percent=np.sum(patch_annotation/255)/patch_annotation.size
                            tumor_percent=0
                            if is_annotated:
                                patch_annotation=annotation_array[x0_patch:x0_patch+patch_size,y0_patch:y0_patch+patch_size]
                                tumor_percent=np.sum(patch_annotation/255)/patch_annotation.size
                            #Si el parche contiene tumor suficiente
                            patch=patch.convert("RGB")
                            if tumor_percent > min_tumor:
                                #save as annotated
                                savedT=savedT+1
                                #print("cancer")
                                patch_name2save=image_folder+'/annotated/'+name+'_xini_'+str(x0_patch)+'_yini_'+str(y0_patch)+'.jpg'
                                patch.save(patch_name2save)
                            elif not is_annotated: #para train, solo guarda sanos de las sanas
                            #else: #este para test 
                                #save as no_annotated
                                savedNA=savedNA+1
                                patch_name2save=image_folder+'/no_annotated/'+name+'_xini_'+str(x0_patch)+'_yini_'+str(y0_patch)+'.jpg'
                                patch.save(patch_name2save)
                            #print(patch_name2save)
                        #plt.title(patch_name2save)
            if savedNA!=0 or savedT!=0:
                print("patches saved in this region: No annotated: "+str(savedNA)+' Annotated: '+str(savedT))
            index=index+1
        
        
        
       
            

# %%




# %%

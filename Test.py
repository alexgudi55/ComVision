# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 15:07:58 2019

@author: Santiago
"""
import csv
import os
import sys
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import zoom
import cv2
import matplotlib.pyplot as plt
import time

def Energy(image):
    fil,col= image.shape
    E=0
    for i in range (fil-1):
        for j in range(col-1):
            E=E+(image[i,j]*image[i,j])
    return E

def readCsv(csvfname):
    # read csv to list of lists
    with open(csvfname, 'r') as csvf:
        reader = csv.reader(csvf)
        csvlines = list(reader)
    return csvlines

def writeCsv(csfname,rows):
    # write csv from list of lists
    with open(csfname, 'w', newline='') as csvf:
        filewriter = csv.writer(csvf)
        filewriter.writerows(rows)
        
def readMhd(filename):
    # read mhd/raw image
    itkimage = sitk.ReadImage(filename)
    scan = sitk.GetArrayFromImage(itkimage) #3D image
    spacing = itkimage.GetSpacing() #voxelsize
    origin = itkimage.GetOrigin() #world coordinates of origin
    transfmat = itkimage.GetDirection() #3D rotation matrix
    return scan,spacing,origin,transfmat

def writeMhd(filename,scan,spacing,origin,transfmat):
    # write mhd/raw image
    itkim = sitk.GetImageFromArray(scan, isVector=False) #3D image
    itkim.SetSpacing(spacing) #voxelsize
    itkim.SetOrigin(origin) #world coordinates of origin
    itkim.SetDirection(transfmat) #3D rotation matrix
    sitk.WriteImage(itkim, filename, False)    

def getImgWorldTransfMats(spacing,transfmat):
    # calc image to world to image transformation matrixes
    transfmat = np.array([transfmat[0:3],transfmat[3:6],transfmat[6:9]])
    for d in range(3):
        transfmat[0:3,d] = transfmat[0:3,d]*spacing[d]
    transfmat_toworld = transfmat #image to world coordinates conversion matrix
    transfmat_toimg = np.linalg.inv(transfmat) #world to image coordinates conversion matrix
    
    return transfmat_toimg,transfmat_toworld

def convertToImgCoord(xyz,origin,transfmat_toimg):
    # convert world to image coordinates
    xyz = xyz - origin
    xyz = np.round(np.matmul(transfmat_toimg,xyz))    
    return xyz
    
def convertToWorldCoord(xyz,origin,transfmat_toworld):
    # convert image to world coordinates
    xyz = np.matmul(transfmat_toworld,xyz)
    xyz = xyz + origin
    return xyz

def extractCube(scan,spacing,xyz,cube_size=80,cube_size_mm=51):
    # Extract cube of cube_size^3 voxels and world dimensions of cube_size_mm^3 mm from scan at image coordinates xyz
    xyz = np.array([xyz[i] for i in [2,1,0]],np.int)
    spacing = np.array([spacing[i] for i in [2,1,0]])
    scan_halfcube_size = np.array(cube_size_mm/spacing/2,np.int)
    if np.any(xyz<scan_halfcube_size) or np.any(xyz+scan_halfcube_size>scan.shape): # check if padding is necessary
        maxsize = max(scan_halfcube_size)
        scan = np.pad(scan,((maxsize,maxsize,)),'constant',constant_values=0)
        xyz = xyz+maxsize
    
    scancube = scan[xyz[0]-scan_halfcube_size[0]:xyz[0]+scan_halfcube_size[0], # extract cube from scan at xyz
                    xyz[1]-scan_halfcube_size[1]:xyz[1]+scan_halfcube_size[1],
                    xyz[2]-scan_halfcube_size[2]:xyz[2]+scan_halfcube_size[2]]

    sh = scancube.shape
    scancube = zoom(scancube,(cube_size/sh[0],cube_size/sh[1],cube_size/sh[2]),order=2) #resample for cube_size
    
    return scancube

def Centroid(a,b,c):
    # Read scan
    [scan,spacing,origin,transfmat] =  readMhd('data/LNDb-{:04}.mhd'.format(a))
    print(spacing,origin,transfmat)
    # Read segmentation mask
    [mask,spacing,origin,transfmat] =  readMhd('masks/LNDb-{:04}_rad{}.mhd'.format(a,b))
    print(spacing,origin,transfmat)
    # Read nodules csv
    csvlines = readCsv('trainNodules.csv')
    header = csvlines[0]
    nodules = csvlines[1:]
    for n in nodules:
        if int(n[header.index('LNDbID')])==a and int(n[header.index('RadID')])==b and int(n[header.index('FindingID')])==c:
            ctr = np.array([float(n[header.index('x')]), float(n[header.index('y')]), float(n[header.index('z')])])
            break
    
    # Convert coordinates to image
    transfmat_toimg,transfmat_toworld = getImgWorldTransfMats(spacing,transfmat)
    print(ctr)
    ctr = convertToImgCoord(ctr,origin,transfmat_toimg)
    print(ctr)
    # Display nodule scan/mask slice
    from matplotlib import pyplot as plt
    fig, axs = plt.subplots(1,2)
    axs[0].imshow(scan[int(ctr[2])])
    axs[1].imshow(mask[int(ctr[2])])
    plt.show()
    return ctr

def ScanMasks(xctr,yctr,zctr,a,b,c,DELTAX,DELTAY,DELTAZ):
    # Read scan
    [scan,spacing,origin,transfmat] =  readMhd('data/LNDb-{:04}.mhd'.format(a))
    #print(spacing,origin,transfmat)
    # Read segmentation mask
    [mask,spacing,origin,transfmat] =  readMhd('masks/LNDb-{:04}_rad{}.mhd'.format(a,b))
    #print(spacing,origin,transfmat)
    # Extract cube around nodule
    scan_cube = extractCube(scan,spacing,[xctr+DELTAZ,yctr+DELTAY,zctr+DELTAX])
    mask[mask!=c] = 0
    mask[mask>0] = 1
    mask_cube = extractCube(mask,spacing,[xctr+DELTAZ,yctr+DELTAY,zctr+DELTAX])
    
    # Display mid slices from resampled scan/mask
    
    fig, axs = plt.subplots(2,3)
    axs[0,0].imshow(scan_cube[int(scan_cube.shape[0]/2),:,:])
    img = mask_cube[int(mask_cube.shape[0]/2),:,:]
    axs[1,0].imshow(mask_cube[int(mask_cube.shape[0]/2),:,:])
    axs[0,1].imshow(scan_cube[:,int(scan_cube.shape[1]/2),:])
    axs[1,1].imshow(mask_cube[:,int(mask_cube.shape[1]/2),:])
    axs[0,2].imshow(scan_cube[:,:,int(scan_cube.shape[2]/2)])
    axs[1,2].imshow(mask_cube[:,:,int(mask_cube.shape[2]/2)])
    #cv2.waitKey(0)   
    time.sleep(5)
#    plt.show()
    fil,col=img.shape
  
    E=0
    for i in range(fil-1):
        for j in range(col-1):
            if ((img[i,j] == 1)):
                E=E+1
    print(E)
    return E

def Volume(xx,yy,zz,MOVEX,MOVEY,MOVEZ):    
    ctr=Centroid(xx,yy,zz)
    E1=ScanMasks(ctr[0],ctr[1],ctr[2],1,1,1,0,0,0)
    if (MOVEX)==1:
        i = 1
        LAYER=1
        while i != 0:
            i=ScanMasks(ctr[0],ctr[1],ctr[2],xx,yy,zz,LAYER,0,0)
            LAYER=LAYER+1
            E1 = E1 + i
    
        i = 1
        LAYER=-1
        while i != 0:
            i=ScanMasks(ctr[0],ctr[1],ctr[2],xx,yy,zz,LAYER,0,0)
            LAYER=LAYER-1
            E1 = E1 + i
    if (MOVEY==1):
        i = 1
        LAYER=1
        while i != 0:
            i=ScanMasks(ctr[0],ctr[1],ctr[2],xx,yy,zz,0,LAYER,0)
            LAYER=LAYER+1
            E1 = E1 + i
    
        i = 1
        LAYER=-1
        while i != 0:
            i=ScanMasks(ctr[0],ctr[1],ctr[2],xx,yy,zz,0,LAYER,0)
            LAYER=LAYER-1
            E1 = E1 + i
        
    if (MOVEZ==1):
        i = 1
        LAYER=1
        while i != 0:
            i=ScanMasks(ctr[0],ctr[1],ctr[2],xx,yy,zz,0,0,LAYER)
            LAYER=LAYER+1
            E1 = E1 + i
    
        i = 1
        LAYER=-1
        while i != 0:
            i=ScanMasks(ctr[0],ctr[1],ctr[2],xx,yy,zz,0,0,LAYER)
            LAYER=LAYER-1
            E1 = E1 + i
    
    return E1

#if __name__ == "__main__":
#    #Extract and display cube for example nodule
#    lnd = 1
#    rad = 1
#    finding = 1
#    # Read scan
#    [scan,spacing,origin,transfmat] =  readMhd('data/LNDb-{:04}.mhd'.format(lnd))
#    print(spacing,origin,transfmat)
#    # Read segmentation mask
#    [mask,spacing,origin,transfmat] =  readMhd('masks/LNDb-{:04}_rad{}.mhd'.format(lnd,rad))
#    print(spacing,origin,transfmat)
#    # Read nodules csv
#    csvlines = readCsv('trainNodules.csv')
#    header = csvlines[0]
#    nodules = csvlines[1:]
#    for n in nodules:
#        if int(n[header.index('LNDbID')])==lnd and int(n[header.index('RadID')])==rad and int(n[header.index('FindingID')])==finding:
#            ctr = np.array([float(n[header.index('x')]), float(n[header.index('y')]), float(n[header.index('z')])])
#            break
#    
#    # Convert coordinates to image
#    transfmat_toimg,transfmat_toworld = getImgWorldTransfMats(spacing,transfmat)
#    print(ctr)
#    ctr = convertToImgCoord(ctr,origin,transfmat_toimg)
#    print(ctr)
#    # Display nodule scan/mask slice
#    from matplotlib import pyplot as plt
#    fig, axs = plt.subplots(1,2)
#    axs[0].imshow(scan[int(ctr[2])])
#    axs[1].imshow(mask[int(ctr[2])])
#    plt.show()
#    
#    # Extract cube around nodule
#    scan_cube = extractCube(scan,spacing,ctr)
#    mask[mask!=finding] = 0
#    mask[mask>0] = 1
#    mask_cube = extractCube(mask,spacing,ctr)
#    
#    # Display mid slices from resampled scan/mask
#    fig, axs = plt.subplots(2,3)
#    axs[0,0].imshow(scan_cube[int(scan_cube.shape[0]/2),:,:])
#    img = mask_cube[int(mask_cube.shape[0]/2),:,:]
#    axs[1,0].imshow(mask_cube[int(mask_cube.shape[0]/2),:,:])
#    axs[0,1].imshow(scan_cube[:,int(scan_cube.shape[1]/2),:])
#    axs[1,1].imshow(mask_cube[:,int(mask_cube.shape[1]/2),:])
#    axs[0,2].imshow(scan_cube[:,:,int(scan_cube.shape[2]/2)])
#    axs[1,2].imshow(mask_cube[:,:,int(mask_cube.shape[2]/2)])    
#    plt.show()

#    H=scan_cube[int(scan_cube.shape[0]/2),:,:]
#    H=H-np.amin(H)
#    H=H/np.amax(H)
#    
#    H2=scan_cube[:,int(scan_cube.shape[1]/2),:]
#    H2=H2-np.amin(H2)
#    H2=H2/np.amax(H2)
#    
#    H3=scan_cube[:,:,int(scan_cube.shape[2]/2)]
#    H3=H3-np.amin(H3)
#    H3=H3/np.amax(H3)
#    
#    G=[Energy(H),Energy(H2),Energy(H3)]
#    print(G)
import numpy as np
import copy
import cv2
import yaml
from matplotlib import pyplot as plt
from utils import readMhd, readCsv, getImgWorldTransfMats, convertToImgCoord, extractCube
from readNoduleList import nodEqDiam
from sklearn.preprocessing import MinMaxScaler


dispFlag = False

def label_img(texture):  
    if texture <= 2 : return [1, 0, 0] 
    elif texture == 3 : return [0, 1, 1] 
    elif texture >3 : return [0, 0, 1]

def normalize(img):
        normalImg = img
        normalImg = normalImg-np.amin(normalImg)
        normalImg =(normalImg/np.amax(normalImg))*255
        normalImg = normalImg.astype(int)
        return normalImg

# Read nodules csv
csvlines = readCsv('trainNodules_gt.csv')
header = csvlines[0]
nodules = csvlines[1:]
lndloaded = -1
findingCount = 0
for n in nodules:
        vol = float(n[header.index('Volume')])
        ctr = np.array([float(n[header.index('x')]), float(n[header.index('y')]), float(n[header.index('z')])])
        lnd = int(n[header.index('LNDbID')])
        rads = list(map(int,list(n[header.index('RadID')].split(','))))
        radfindings = list(map(int,list(n[header.index('RadFindingID')].split(','))))
        finding = int(n[header.index('FindingID')])
        noduleTexture = (n[header.index('Text')])
        noduleTexture = float(n[header.index('Text')])
        #label = label_img(noduleTexture)
        print(lnd,finding,rads,radfindings,noduleTexture)
        #print("nodule Texture", label)
        # Read scan
        if lnd!=lndloaded:
                [scan,spacing,origin,transfmat] =  readMhd('data/LNDb-{:04}.mhd'.format(lnd))                
                transfmat_toimg,transfmat_toworld = getImgWorldTransfMats(spacing,transfmat)
                lndloaded = lnd
        
        # Convert coordinates to image
        ctr = convertToImgCoord(ctr,origin,transfmat_toimg)                
        
        #for rad,radfinding in zip(rads,radfindings):
        # Read segmentation mask
        #[mask,_,_,_] =  readMhd('masks/LNDb-{:04}_rad{}.mhd'.format(lnd,rad))
        
        # Extract cube around nodule
        scan_cube = extractCube(scan,spacing,ctr)
        """masknod = copy.copy(mask)
        masknod[masknod!=radfinding] = 0
        masknod[masknod>0] = 1
        mask_cube = extractCube(masknod,spacing,ctr)"""
        
        slice1 = scan_cube[int(scan_cube.shape[0]/2),:,:]
        slice2 = scan_cube[:,int(scan_cube.shape[1]/2),:]
        slice3 = scan_cube[:,:,int(scan_cube.shape[2]/2)]
        # Display mid slices from resampled scan/mask
        if dispFlag:
                fig, axs = plt.subplots(2,3)
                axs[0,0].imshow(slice1)
                axs[1,0].imshow(mask_cube[int(mask_cube.shape[0]/2),:,:])
                axs[0,1].imshow(slice2)
                axs[1,1].imshow(mask_cube[:,int(mask_cube.shape[1]/2),:])
                axs[0,2].imshow(slice3)
                axs[1,2].imshow(mask_cube[:,:,int(mask_cube.shape[2]/2)])    
                plt.show()
        slice1 = normalize(slice1)
        slice2 = normalize(slice2)
        slice3 = normalize(slice3)
        path = str ()
        if noduleTexture < 3:
                path = 'GGO'
        elif noduleTexture == 3:
                path = 'partSolid'
        elif noduleTexture > 3:
                path = 'solid'
        cv2.imwrite("slices/"+path+"/LNDb-{:04d}_finding{}_s1.jpg".format(lnd,finding),slice1)
        cv2.imwrite("slices/"+path+"/LNDb-{:04d}_finding{}_s2.jpg".format(lnd,finding),slice2)
        cv2.imwrite("slices/"+path+"/LNDb-{:04d}_finding{}_s3.jpg".format(lnd,finding),slice3)           
                # Save mask cubes
                #print(mask_cube.shape)
                #np.save('mask_cubes/'+path+'/LNDb-{:04d}_finding{}_rad{}.jpg'.format(lnd,finding,rad),mask_cube)            
                

# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 16:32:29 2020

@author: 
"""

import cv2

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

img = Image.open('lssd50.jpg')
img=np.asarray(img)
print("Newimg",img)

#imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure()
plt.imshow(img)
#print("Image",img)
Ycrcbimg=cv2.cvtColor(img,cv2.COLOR_RGB2YCR_CB)
print("Ycrcb_img",Ycrcbimg)
print()
Y_channel=Ycrcbimg[:,:,0]
print(Y_channel.shape)
Y_mean=np.mean(Y_channel,axis=(0,1))
print(Y_mean)
plt.figure()
plt.subplot(1,2,1)
plt.imshow(Y_channel)
hist,bins = np.histogram( Y_channel.flatten(),256,[0,256])
plt.subplot(1,2,2)
plt.hist( Y_channel.flatten(),256,[0,256])
#colormap(grey)
cdf = hist.cumsum()
cdf_normalized = cdf * hist.max()/ cdf.max()
plt.plot(cdf_normalized, color = 'r')
cdf_m = np.ma.masked_equal(cdf,0)
#Histogram equalization
cdf_m = np.ma.masked_equal(cdf,0)
cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
cdf = np.ma.filled(cdf_m,0).astype('uint8')

Y_equalizedim=cdf[Y_channel]
hist2,bins2 = np.histogram( Y_equalizedim.flatten(),256,[0,256])
plt.figure()
plt.subplot(1,2,1)
plt.imshow(Y_equalizedim)
plt.subplot(1,2,2)
plt.hist(Y_equalizedim.flatten(),256,[0,256])
cdfequ=hist2.cumsum()
cdfequ_normalized=cdfequ*hist2.max()/cdfequ.max()
plt.plot(cdfequ_normalized,color='r')

Y_equalizedim_mean=np.mean(Y_equalizedim,axis=(0,1))
print("Mean2",Y_equalizedim_mean)
#Shadow masking
mas=np.empty([Y_channel.shape[0],Y_channel.shape[1]],dtype=int)
mas1=np.empty([Y_channel.shape[0],Y_channel.shape[1]],dtype=int)
#print(Y_equalizedim)
mas[(Y_equalizedim<0.52*Y_equalizedim_mean)]=4
mas[(Y_equalizedim>=0.52*Y_equalizedim_mean)]=255

#plt.figure()
#plt.imshow(mas)
kernel = np.ones((3,3),np.float32)/9
filimg=cv2.filter2D(Y_equalizedim,-1,kernel)
print("Equim",Y_equalizedim)
print("filteredim",filimg)
mas1[(mas<=0.7*filimg)]=0
mas1[(mas>0.7*filimg)]=1
print(mas1)
sdmask=np.empty(img.shape)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if(mas1[i,j]==0):
            sdmask[i,j,:]=1
        else:
            sdmask[i,j,:]=0
#sdmask=cv2.medianBlur(sdmask,5)
plt.figure()
plt.imshow(sdmask)

print()
#Shadow removal
k=mas1
print("kk",k)

Rcha=img[:,:,0]
litr=Rcha[(mas1==1)]
sr=Rcha[(mas1==0)]
litr=np.reshape(litr,(len(litr),1))
sr=np.reshape(sr,(len(sr),1))
litrAvg=np.mean(litr,axis=0)
srAvg=np.mean(sr,axis=0)
print(litrAvg,srAvg)
ratioR=litrAvg/srAvg

Gcha=img[:,:,1]
litg=Gcha[(mas1==1)]
sg=Gcha[(mas1==0)]
litg=np.reshape(litg,(len(litg),1))
sg=np.reshape(sg,(len(sg),1))
litgAvg=np.mean(litg,axis=0)
sgAvg=np.mean(sg,axis=0)
print(litgAvg,sgAvg)
ratioG=litgAvg/sgAvg

Bcha=img[:,:,2]
litb=Bcha[(mas1==1)]
sb=Bcha[(mas1==0)]
litb=np.reshape(litb,(len(litb),1))
sb=np.reshape(sb,(len(sb),1))
litbAvg=np.mean(litb,axis=0)
sbAvg=np.mean(sb,axis=0)
print(litbAvg,sbAvg)
ratioB=litbAvg/sbAvg

print("averages")
print(ratioR,ratioG)
print("Shadow free image")

imgshadowfree=np.copy(img)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if(mas1[i,j]==0):
            imgshadowfree[i,j,0]=img[i,j,0]+(int)(litrAvg-srAvg)
            imgshadowfree[i,j,1]=img[i,j,1]+(int)(litgAvg-sgAvg)
            imgshadowfree[i,j,2]=img[i,j,2]+(int)(litbAvg-sbAvg)
        elif(mas1[i,j]==1):
            imgshadowfree[i,j,0]=img[i,j,0]
            imgshadowfree[i,j,1]=img[i,j,1]
            imgshadowfree[i,j,2]=img[i,j,2]
            

print(imgshadowfree)            
    
#imr=Image.fromarray(imgshadowfree[:,:,2],mode=None)
#img=Image.fromarray(imgshadowfree[:,:,1],mode=None)
#imb=Image.fromarray(imgshadowfree[:,:,0],mode=None)        
#imgshfree=cv2.merge((imgshadowfree[:,:,0],imgshadowfree[:,:,1],imgshadowfree[:,:,2]))        
#print("Imageshadowfree",imgshfree.shape)
imgres=Image.fromarray(imgshadowfree)
imgres.save('shawdwfree.jpg')
shadowfreeimg=Image.open('shawdwfree.jpg')
shadowfreeimg=np.asarray(shadowfreeimg)
print("dsdjsd",shadowfreeimg.shape)
filteredshadowfreeimg=cv2.medianBlur(shadowfreeimg,5)
plt.figure()
plt.imshow(imgres)




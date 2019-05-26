import ImageIO as image 
import cv2.cv2 as cv2
import numpy as np 
import matplotlib.pyplot as plt

reader=image.reader()

def mask_generator(size,x,y,r):
    mask=np.zeros(size,dtype=np.uint8)
    mask[max(x-r,0):min(x+r,size[0]-1),max(y-r,0):min(y+r,size[1]-1)]=255
    return mask

if __name__ == "__main__":
    mask=mask_generator(reader.size,100,100,50)
    cv2.imshow("mask",mask)
    _,a=reader.read()
    addresult=cv2.add(a,np.zeros(np.shape(a),dtype=np.uint8),mask=mask)
    cv2.imshow("add",addresult)
    cv2.waitKey()
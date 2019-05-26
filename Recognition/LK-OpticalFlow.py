import ImageIO as image 
import cv2.cv2 as cv2
import numpy as np 
import matplotlib.pyplot as plt

def Mask_Square(size,x,y,r):
    mask=np.zeros(size,dtype=np.uint8)
    mask[max(x-r,0):min(x+r,size[0]-1),max(y-r,0):min(y+r,size[1]-1)]=255
    return mask

def Feature(image,mask):
    feature=cv2.goodFeaturesToTrack(image,10,0.3,5,mask=mask)
    return feature

def Optical_Flow(image,p0,mask=None):
    lk_params = dict(
        winSize  = (15,15),
        maxLevel = 2,
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    image_old=image
    image_old=cv2.add(image_old,np.zeros(np.shape(image_old),dtype=np.uint8),mask=mask)
    linemask = np.zeros_like(image)
    while 1:
        image=cv2.add(image,np.zeros(np.shape(image),dtype=np.uint8),mask=mask)
        p1, st, err = cv2.calcOpticalFlowPyrLK(image_old, image, p0, None, **lk_params)
        good_new = p1[st==1]
        good_old = p0[st==1]
        for i,(new,old) in enumerate(zip(good_new, good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            linemask = cv2.line(linemask, (a,b),(c,d), color[i].tolist(), 2)
            image = cv2.circle(image,(a,b),5,color[i].tolist(),-1)
        img = cv2.add(image,linemask)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        # Now update the previous frame and previous points
        image_old = image.copy()
        p0 = good_new.reshape(-1,1,2)
        image=(yield img)



if __name__ == "__main__":
    reader=image.reader()

    mask=Mask_Square(reader.size,300,300,100)
    #cv2.imshow("mask",mask)

    _,a=reader.read()
    _,b=reader.read(False)
    cv2.imshow("s",a)
    addresult=cv2.add(a,np.zeros(np.shape(a),dtype=np.uint8),mask=mask)
    #cv2.imshow("add",addresult)

    feat=Feature(a,mask)

    color=np.random.randint(0,255,[50,3])
    for i,point in enumerate(feat):
        b = cv2.circle(b,(point[0][0],point[0][1]),5,color[i].tolist(),-1)

    print(feat)
    #cv2.imshow("with feature",b)

    _,currentimage=reader.read()
    gen=Optical_Flow(currentimage,feat,mask=mask)
    next(gen)
    while(1):
        _,currentimage=reader.read()
        feedback=gen.send(currentimage)
        cv2.imshow("opticalflow",feedback)

    cv2.waitKey()
    cv2.destroyAllWindows()

    



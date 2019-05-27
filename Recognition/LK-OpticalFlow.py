import ImageIO as image 
import cv2.cv2 as cv2
import numpy as np 
import matplotlib.pyplot as plt
from warnings import warn
import argparse
from time import sleep

def Mask_Square(size,x,y,r):
    mask=np.zeros(size,dtype=np.uint8)
    mask[max(x-r,0):min(x+r,size[0]-1),max(y-r,0):min(y+r,size[1]-1)]=255
    return mask

def Feature(image,mask):
    feature=cv2.goodFeaturesToTrack(image,50,0.1,5,mask=mask)
    return feature

def LK_Optical_Flow(image,p0,mask=None):
    lk_params = dict(
        winSize  = (50,50),
        maxLevel = 5,
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    image_old=image
    image_old=cv2.add(image_old,np.zeros(np.shape(image_old),dtype=np.uint8),mask=mask)
    linemask = np.zeros_like(image)
    while 1:
        #image=cv2.add(image,np.zeros(np.shape(image),dtype=np.uint8),mask=mask)
        p1, st, err = cv2.calcOpticalFlowPyrLK(image_old, image, p0, None, **lk_params)
        try:
            good_new = p1[st==1]
            good_old = p0[st==1]
        except TypeError:
            warn("Lose track")
            return
        for i,(new,old) in enumerate(zip(good_new, good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            linemask = cv2.line(linemask, (a,b),(c,d), color[i].tolist(), 2)
            image = cv2.circle(image,(a,b),5,color[i].tolist(),-1)
        img = cv2.add(image,linemask)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        image_old = image.copy()
        p0 = good_new.reshape(-1,1,2)
        image=(yield img)

def Dense_Optical_Flow(image):
    image_old=image
    frame1=image
    next=image
    hsv = np.zeros([frame1.shape[0],frame1.shape[1],3],dtype=np.uint8)
    hsv[...,1] = 255
    while(1):
        flow = cv2.calcOpticalFlowFarneback(image_old,next, None, 0.5, 3, 50, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        image_old = next
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        next=(yield bgr)

def Diff_Analyze(image_init):
    '''
    图片差异分析，检测创建时的帧和之后输入的帧之间的差距
    '''
    firstFrame = image_init
    image=image_init

    while True:
        # 创建参数解析器并解析参数
        ap = argparse.ArgumentParser()
        ap.add_argument("-v", "--video", help="path to the video file")
        ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
        args = vars(ap.parse_args())

        frame=image
        
        # 调整该帧的大小，转换为灰阶图像并且对其进行高斯模糊
        # frame = imutils.resize(frame, width=500)
        gray = frame
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # 计算当前帧和第一帧的不同
        frameDelta = cv2.absdiff(firstFrame, gray)
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
    
        # 扩展阀值图像填充孔洞，然后找到阀值图像上的轮廓
        #thresh = cv2.dilate(thresh, None, iterations=2)
        (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
    
        # 遍历轮廓
        for c in cnts:
            # if the contour is too small, ignore it
            if cv2.contourArea(c) < args["min_area"]:
                continue
    
            # 计算轮廓的边界框，在当前帧中画出该框
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            

        #显示当前帧并记录用户是否按下按键
        cv2.imshow("Security Feed", frame)
        cv2.imshow("Thresh", thresh)
        cv2.imshow("Frame Delta", frameDelta)
        
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        
        image=(yield thresh)

if __name__ == "__main__":
    reader=image.reader()

    mask=Mask_Square(reader.size,100,100,50)
    #cv2.imshow("mask",mask)

    _,a=reader.read()
    _,b=reader.read(False)
    #cv2.imshow("s",a)
    addresult=cv2.add(a,np.zeros(np.shape(a),dtype=np.uint8),mask=mask)
    #cv2.imshow("add",addresult)

    feat=Feature(a,mask)

    color=np.random.randint(0,255,[50,3])
    for i,point in enumerate(feat):
        b = cv2.circle(b,(point[0][0],point[0][1]),5,color[i].tolist(),-1)

    print(feat)
    #cv2.imshow("with feature",b)

    _,currentimage=reader.read()
#    gen=LK_Optical_Flow(currentimage,feat,mask=mask)
#    gen=Dense_Optical_Flow(currentimage)
    gen=Diff_Analyze(currentimage)
    next(gen)
    while(1):
        #sleep(1)
        _,currentimage=reader.read()
        feedback=gen.send(currentimage)
        cv2.imshow("opticalflow",feedback)
        percentage=np.sum(feedback)/(254*480*640)
        print(percentage)

    cv2.waitKey()
    cv2.destroyAllWindows()

    



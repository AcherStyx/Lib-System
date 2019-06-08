import ImageIO as image
import cv2.cv2 as cv2
import numpy as np 
import matplotlib.pyplot as plt
from warnings import warn
import argparse
from time import sleep
import pylab
import imutils
import CSVToFile as fileio
import time

def Mask_Square(shape,x,y,r):
    mask=np.zeros(shape,dtype=np.uint8)
    mask[max(x-r,0):min(x+r,shape[0]-1),max(y-r,0):min(y+r,shape[1]-1)]=255
    return mask

def Mask_Polygon(shape,pointlist):
    mask=np.zeros(shape, dtype = "uint8")
    #mask=cv2.polylines(mask, pointlist, 1, 255)
    mask=cv2.fillPoly(mask, pointlist, 255)
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

class Diff_Analyzer:
    '''
    图片差异分析
    '''
    __init_frame=None
    def __init__(self,image_init):
        self.__init_frame=image_init
    def change(self,image,mode=0,gaussian=5,valve=100):
        '''
        mode        return
        0       带阀值限制后的输出
        1       未使用阀值过滤的原始差异结果
        2       带边框指示变化区域的原始图

        gaussian    高斯模糊的程度
        valve       过滤阀值
        '''
        # 创建参数解析器并解析参数
        ap = argparse.ArgumentParser()
        ap.add_argument("-v", "--video", help="path to the video file")
        ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
        args = vars(ap.parse_args())
        
        # 高斯模糊，防噪点
        image=cv2.GaussianBlur(image,(gaussian,gaussian),0)

        # 计算当前帧和第一帧的不同
        frameDelta = cv2.absdiff(self.__init_frame, image)
        thresh = cv2.threshold(frameDelta, valve, 255, cv2.THRESH_BINARY)[1]
    
        if mode==0:
            #cv2.imshow("Thresh", thresh)
            return thresh
        elif mode==1:
            #cv2.imshow("Frame Delta", frameDelta)
            return frameDelta
        elif mode==2:
            # 创建边框
            (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                if cv2.contourArea(c) < args["min_area"]:
                    continue
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
            # 输出带矩形标记的当前帧
            cv2.imshow("Security Feed", image)
            return image
        else:
            raise(ValueError)

def Useage(image,mask,times=1):
    sum_of_mask=np.sum(mask)
    masked=cv2.bitwise_and(image, image, mask=mask)
    sum_of_image=np.sum(masked)
    return sum_of_image/sum_of_mask*times

def Useage_Filter(list):
    out=[]
    valve=10
    valve2=20
    for elem in list:
        if valve2>=elem>=valve:
            out.append("物品占座")
        elif elem<valve:
            out.append("空闲")
        else:
            out.append("使用中")
    return out

def GetCapturePointList(image,n):    #image为所要交互获取点的图像，n为想要获取的点的个数
    im_array = np.array(image)
    pylab.imshow(im_array)
    X = pylab.ginput(n)  # x 为列表，元素为坐标 例：[（x1,y1),(x2,y2),......]
    pylab.close()
    return X

if __name__ == "__main__":
    import imutils

    reader=image.reader("E:/[工程项目]/[数据集]/[图书馆占座识别视频样本]/00129_1.mp4")
    _,a=reader.read()
    _,b=reader.read(False)

    print("="*10,"蒙版","="*10)    
#    mask=Mask_Square(reader.size,100,100,50)
    pointlist=np.array([[242,351],[407,353],[398,478],[213,479]], dtype = np.int32)
    pointlist=pointlist.reshape([1,4,2])
    mask=Mask_Polygon(a.shape,pointlist)
    #cv2.imshow("mask",mask)
    #cv2.imshow("s",a)
    addresult=cv2.add(a,np.zeros(np.shape(a),dtype=np.uint8),mask=mask)
    #cv2.imshow("add",addresult)

    print("="*10,"特征点提取","="*10)
    feat=Feature(a,mask)
    color=np.random.randint(0,255,[50,3])
    for i,point in enumerate(feat):
        b = cv2.circle(b,(point[0][0],point[0][1]),5,color[i].tolist(),-1)
    print(feat)
    #cv2.imshow("with feature",b)


    print("="*10,"差异分析","="*10)

    
    seat1=[[307,670],[671,611],[869,694],[905,954]]
    seat2=[[313,673],[716,614],(853, 503), (617, 449)]
    seat3=[[(625, 450), (831, 517), (970, 424), (756, 360)]]
    seat4=[[(759, 359), (972, 427), (1080, 314), (1069, 173)]]
    seat5=[[(1077, 175), (1073, 300), (1184, 354), (1495, 267)]]
    seat6=[[(1497, 269), (1194, 339), (1080, 426), (1344, 494)]]
    seat7=[[(1357, 466), (1106, 394), (960, 536), (1209, 670)]]
    seat8=[[(1180, 694), (939, 576), (846, 687), (903, 956)]]
    seats=[seat1]+[seat2]+[seat3]+[seat4]+[seat5]+[seat6]+[seat7]+[seat8]
    # 打印第一张作为示例
    _,currentimage=reader.read(False)
    #currentimage = imutils.resize(currentimage, width=300)
    #cv2.imshow("Init",currentimage)
    cv2.waitKey()
    
    #seats=[]
    #for i in range(3):
    #    seat=GetCapturePointList(currentimage,4)
    #    cv2.waitKey()
    #    seats.append(seat)
    #    print(seat)
    #seats=np.array(seats)
    #seats.reshape([3,1,3,2])
        

    # 开始计算差异
    _,currentimage=reader.read()
    #currentimage = imutils.resize(currentimage, width=800)
    # 建立MASK
    mask=[]
    for index,seat in enumerate(seats):    
        pointlist=np.array(seat, dtype = np.int32)
        pointlist=pointlist.reshape([1,-1,2])
        mask.append(Mask_Polygon(currentimage.shape,pointlist)) 
        #cv2.imshow("Mask{i}".format(i=index),mask[index])

#    gen=LK_Optical_Flow(currentimage,feat,mask=mask)
#    gen=Dense_Optical_Flow(currentimage)
#    gen=Diff_Analyze(currentimage)
    gen=Diff_Analyzer(currentimage)
#    next(gen)
    i=0
    while(1):
        # sleep(0.1)
        
        _,currentimage=reader.read()
        if i%100==0:
            i+=1
        else:
            i+=1
            continue

        currentimage2 = imutils.resize(currentimage, width=800)
        cv2.imshow("Current",currentimage2)
#        feedback=gen.send(currentimage)
        feedback=gen.change(currentimage,mode=0,valve=25)        
        cv2.waitKey(30)
        infotowrite=[]
        for index,seatmask in enumerate(mask):
            percentage=Useage(feedback,seatmask,100)
            filted=Useage_Filter([percentage])
            print("ID: {index} 占用率: {use:05.2f} 情况判断: {situ}".format(index=index,use=percentage,situ=filted[0]))
            infotowrite.append([index,percentage])
        feedback = imutils.resize(feedback, width=800)
        cv2.imshow("DiffAnalyze",feedback)
        fileio.CSVToFile([[time.ctime()]]+infotowrite,"./Recognition/.sampleoutput/{index}.txt".format(index=i))
        #cv2.imwrite("./Recognition/.sampleoutput/{index}.jpg".format(index=i),currentimage2)
        print("\n")

    cv2.waitKey()   
    cv2.destroyAllWindows()

    



import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
kernel = np.ones((3,3),np.uint8)


img = cv.imread('test7.jpg')
print(img.shape)

#BGR转HSV
hsvimg = cv.cvtColor(img,cv.COLOR_BGR2HSV)

#设置阈值，蓝色
lowblue = np.array([100,43,100])
highblue = np.array([124,255,255])

#掩膜操作，将阈值内的区域置255，阈值外的置0,得到蓝色部分
blue = cv.inRange(hsvimg, lowblue, highblue)
BlueThings = cv.bitwise_and(img, img, mask=blue)

#处理图片
gray = cv.cvtColor(BlueThings,cv.COLOR_BGR2GRAY)
threshold = cv.threshold(gray,1,255,cv.THRESH_BINARY)[1]
imgclose = cv.morphologyEx(threshold,cv.MORPH_CLOSE,kernel,iterations=3)

#得到轮廓
cnt = cv.findContours(imgclose,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)[0]

#对检测到的轮廓通过面积排序
conts = sorted(cnt,key=cv.contourArea,reverse=True)
print(len(cnt))

#获得外接矩形四个顶点
n = cv.minAreaRect(conts[0])
box = cv.boxPoints(n)
box = np.array(box,np.int32)
# cv.polylines(img,[box],True,(0,255,255),3)#通过顶点坐标画出矩形

#x小的两个放前面
sort = box
ave_x = int((sort[0][0] + sort[1][0] + sort[2][0] + sort[3][0])/4)
sort_x = []
print(sort)
for i in sort:
    if i[0] > ave_x:
        sort_x.append(i)
    else:
        sort_x.insert(0,i)

#排y的顺序
if sort_x[0][1] < sort_x[1][1]:
    sort_x[0],sort_x[1] = sort_x[1],sort_x[0]

if sort_x[2][1] > sort_x[3][1]:
    sort_x[2], sort_x[3] = sort_x[3], sort_x[2]

print(sort_x)

#要变换成的坐标
box32 = np.float32(sort_x)   #先转换为浮点型
target = np.array([
    [0,120],
    [0,0],
    [400,0],
    [400,120]],np.float32)

#求变换矩阵
M = cv.getPerspectiveTransform(box32,target)

#对原图进行变换
change = cv.warpPerspective(img,M,(400,120))


plt.imshow(change)
plt.show()
cv.imshow('100',change)
cv.waitKey(0)
cv.destroyAllWindows()
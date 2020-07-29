import cv2 as cv
import numpy as np
kernel = np.ones((3,3),np.uint8)

#读入图片
img = cv.imread('test5.jpg')
cv.imshow('1',img)
########################################################################################

#双边滤波
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
lvbo = cv.bilateralFilter(gray,5,75,75)
cv.imshow('2',lvbo)
########################################################################################

#canny
canny = cv.Canny(lvbo,400,600)
cv.imshow('3',canny)
########################################################################################

#获得轮廓并筛选
cnts = cv.findContours(canny,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)[0]
# cv.drawContours(img,cnts,-1,(255,0,0),2)
fits = []
for cnt in cnts:
    x,y,w,h = cv.boundingRect(cnt)
    if 3 < w/h < 3.3:
        fits.append([x,y,w,h])

for fit in fits:
    rect = cv.rectangle(img.copy(),(fit[0],fit[1]),(fit[0]+fit[2],fit[1]+fit[3]),(0,0,255),2)

cv.imshow('4',rect)
########################################################################################

#取出车牌部分
x,y,w,h = fits[0]
plate_part = img[y:y+h,x:x+w]
cv.imshow('5',plate_part)
########################################################################################

#对车牌部分提取轮廓
gray_plate = cv.cvtColor(plate_part,cv.COLOR_BGR2GRAY)
thre = cv.threshold(gray_plate,127,255,cv.THRESH_BINARY)[1]
cnts_plate = cv.findContours(thre,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)[0]

if len(cnts_plate) == 1:
    cnt_plate = cnts_plate[0]

draw_cnts = cv.drawContours(plate_part.copy(),cnt_plate,-1,(0,0,255),1)
cv.imshow('6',draw_cnts)
########################################################################################

#取轮廓的最小外界矩形，为仿射变换做准备
n = cv.minAreaRect(cnt_plate)
box = cv.boxPoints(n)
box = np.array(box,np.int32)
# minrect = cv.polylines(plate_part.copy(),[box],True,(0,255,255),1)#通过顶点坐标画出矩形
########################################################################################

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
print(sort_x)#打印顶点，顺序为左下、左上、右上、右下
########################################################################################

#仿射变换
#要变换成的坐标
box32 = np.float32(sort_x)   #先转换为浮点型
target = np.array([
    [0,140],
    [0,0],
    [440,0],
    [440,140]],np.float32)

#求变换矩阵
M = cv.getPerspectiveTransform(box32,target)

#对原图进行变换
result = cv.warpPerspective(plate_part,M,(440,140))
cv.imshow('7',result)
#########################################################################################

cv.waitKey(0)
cv.destroyAllWindows()
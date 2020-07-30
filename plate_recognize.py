import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pickle
from skimage.measure import compare_ssim

kernel = np.ones((3,3),np.uint8)
plate_final = []


#初始化模板
moban_file = open('moban_list','rb')
moban_list = pickle.load(moban_file)
name_file = open('name_list','rb')
name_list = pickle.load(name_file)


#读入图片
plate = cv.imread('test.png')
gray = cv.cvtColor(plate,cv.COLOR_BGR2GRAY)

#汉字部分寻找
_,erzhi1 = cv.threshold(gray,127,255,cv.THRESH_BINARY)
imgopen = cv.morphologyEx(erzhi1,cv.MORPH_OPEN,kernel,iterations=1)
imgclose = cv.morphologyEx(imgopen,cv.MORPH_CLOSE,kernel,iterations=3)
cv.imshow('imgclose',imgclose)#############################################
contours_target2,_2 = cv.findContours(imgclose,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)

#筛选外接矩形，得到汉字部分
for i in range(len(contours_target2)):
    x1,y1,w1,h1 = cv.boundingRect(contours_target2[i])
    if h1/w1 >1.5 and x1 < erzhi1.shape[1]/8:
        # imgrect = cv.rectangle(plate, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)
        provin_name = erzhi1[y1-int(0.06*h1):y1+h1+int(0.06*h1),x1-int(0.06*w1):x1+w1+int(0.06*w1)]
        provin_name = cv.resize(provin_name,(76,76))
cv.imshow('zi',provin_name)

#汉字模板匹配
similar_list1 = []
for i in range(len(name_list)):
    (score1, diff) = compare_ssim(provin_name, name_list[i], full=True)
    similar_list1.append(score1)
index_max1 = similar_list1.index(max(similar_list1))
if index_max1 == 0:
    plate_final.append('台')
if index_max1 == 1:
    plate_final.append('澳')
if index_max1 == 2:
    plate_final.append('港')
if index_max1 == 3:
    plate_final.append('新')
if index_max1 == 4:
    plate_final.append('青')
if index_max1 == 5:
    plate_final.append('宁')
if index_max1 == 6:
    plate_final.append('甘')
if index_max1 == 7:
    plate_final.append('陕')
if index_max1 == 8:
    plate_final.append('藏')
if index_max1 == 9:
    plate_final.append('渝')
if index_max1 == 10:
    plate_final.append('川')
if index_max1 == 11:
    plate_final.append('贵')
if index_max1 == 12:
    plate_final.append('云')
if index_max1 == 13:
    plate_final.append('桂')
if index_max1 == 14:
    plate_final.append('粤')
if index_max1 == 15:
    plate_final.append('湘')
if index_max1 == 16:
    plate_final.append('鄂')
if index_max1 == 17:
    plate_final.append('豫')
if index_max1 == 18:
    plate_final.append('鲁')
if index_max1 == 19:
    plate_final.append('闽')
if index_max1 == 20:
    plate_final.append('赣')
if index_max1 == 21:
    plate_final.append('皖')
if index_max1 == 22:
    plate_final.append('浙')
if index_max1 == 23:
    plate_final.append('苏')
if index_max1 == 24:
    plate_final.append('沪')
if index_max1 == 25:
    plate_final.append('辽')
if index_max1 == 26:
    plate_final.append('京')
if index_max1 == 27:
    plate_final.append('吉')
if index_max1 == 28:
    plate_final.append('黑')
if index_max1 == 29:
    plate_final.append('蒙')
if index_max1 == 30:
    plate_final.append('晋')
if index_max1 == 31:
    plate_final.append('冀')
if index_max1 == 32:
    plate_final.append('津')




# o = cv.drawContours(plate.copy(),contours_target2,-1,(0,0,255),1)
# cv.imshow('2',provin_name)

#数字字母部分寻找
_,erzhi2 = cv.threshold(gray,127,255,cv.THRESH_BINARY)
erode2 = cv.erode(erzhi2,kernel,iterations=2)
dilate2 = cv.dilate(erode2,kernel,iterations=2)
contours_target,_ = cv.findContours(dilate2,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
# o = cv.drawContours(plate.copy(),contours_target,-1,(0,0,255),1)
# cv.imshow('2',o)

#选出数字部分并对各个字母数字排序
all_x = []
for n in range(len(contours_target)):
    x, y, w, h = cv.boundingRect(contours_target[n])
    if h/w > 1.5:
        all_x.append(x)
all_x_reset = sorted(all_x)
all_x_reset.insert(0,0)
#得到车牌样本
all_target = []
for i in range(len(all_x)):
    for m in range(len(contours_target)):
        x,y,w,h = cv.boundingRect(contours_target[m])
        if h/w > 1.5:
            a = max(all_x_reset)
            if a-1 < x < a + 1:

                target = gray[y:y+h,x:x+w]
                _, target_erzhi = cv.threshold(target, 127, 255, cv.THRESH_BINARY)
                target_resize = cv.resize(target_erzhi,(18,36))
                all_target.insert(0,target_resize)
                all_x_reset.pop()

#对样本各个字母进行模板匹配，得到最终结果

for i in range(len(all_target)):
    similar_list = []
    for m in range(len(moban_list)):
        (score, diff) = compare_ssim(all_target[i], moban_list[m], full=True)
        similar_list.append(score)
    index_max = similar_list.index(max(similar_list))
    if index_max == 0:
        plate_final.append('A')
    if index_max == 1:
        plate_final.append('B')
    if index_max == 2:
        plate_final.append('C')
    if index_max == 3:
        plate_final.append('D')
    if index_max == 4:
        plate_final.append('E')
    if index_max == 5:
        plate_final.append('F')
    if index_max == 6:
        plate_final.append('G')
    if index_max == 7:
        plate_final.append('H')
    if index_max == 8:
        plate_final.append(1)
    if index_max == 9:
        plate_final.append('J')
    if index_max == 10:
        plate_final.append('K')
    if index_max == 11:
        plate_final.append('L')
    if index_max == 12:
        plate_final.append('M')
    if index_max == 13:
        plate_final.append('N')
    if index_max == 14:
        plate_final.append(0)
    if index_max == 15:
        plate_final.append('P')
    if index_max == 16:
        plate_final.append('Q')
    if index_max == 17:
        plate_final.append('R')
    if index_max == 18:
        plate_final.append('S')
    if index_max == 20:
        plate_final.append('T')
    if index_max == 21:
        plate_final.append('U')
    if index_max == 22:
        plate_final.append('V')
    if index_max == 23:
        plate_final.append('W')
    if index_max == 24:
        plate_final.append('X')
    if index_max == 25:
        plate_final.append('Y')
    if index_max == 26:
        plate_final.append('Z')
    if index_max == 27:
        plate_final.append(0)
    if index_max == 28:
        plate_final.append(1)
    if index_max == 29:
        plate_final.append(2)
    if index_max == 30:
        plate_final.append(3)
    if index_max == 31:
        plate_final.append(4)
    if index_max == 32:
        plate_final.append(5)
    if index_max == 19:
        plate_final.append(6)
    if index_max == 33:
        plate_final.append(7)
    if index_max == 34:
        plate_final.append(8)
    if index_max == 35:
        plate_final.append(9)

#输出结果
print('目标车牌为：',plate_final)


cv.imshow('原图',plate)
# plt.imshow(erzhi)
# plt.show()

if cv.waitKey(0)==27:
    cv.destroyAllWindows()


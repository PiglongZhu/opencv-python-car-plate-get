import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import skimage.measure
import pickle
kernel = np.ones((3,3),np.uint8)

img = cv.imread('shengname.jpg')
# gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# _,erzhi = cv.threshold(gray,127,255,cv.THRESH_BINARY_INV)
# dilate = cv.dilate(erzhi,kernel,iterations=3)
# #取轮廓
# contours,hierarchy = cv.findContours(dilate,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
# #画轮廓
# img_copy = img.copy()
# img_contours = cv.drawContours(img_copy,contours,-1,(0,0,255),2)
#
# #画外接矩形
# all_name = []
# for i in range(len(contours)):
#     x,y,w,h = cv.boundingRect(contours[i])
#     # imgrect = cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#     name_cut = erzhi[y:y+h,x:x+w]
#     name_cut = cv.resize(name_cut,(76,76))
#     all_name.append(name_cut)

# name_list = open(r'D:\pycharm\plate_recognize\name_list','wb')
# pickle.dump(all_name,name_list)
# name_list.close()

moban_file = open('name_list','rb')
all_name = pickle.load(moban_file)


# cv.imshow('1',all_name[31])
# plt.imshow(all_name[2])
# plt.show()

if cv.waitKey(0) == 27:
    cv.destroyAllWindows()
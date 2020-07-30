import pickle
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
kernel = np.ones((3,3),np.uint8)

moban = cv.imread('moban2.jpg')
moban_gray = cv.cvtColor(moban,cv.COLOR_BGR2GRAY)
_,moban_erzhi = cv.threshold(moban_gray,127,255,cv.THRESH_BINARY)
# openimg = cv.morphologyEx(moban_erzhi,cv.MORPH_OPEN,kernel)
contours,_ = cv.findContours(moban_erzhi,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
list = []
for i in range(len(contours)):
    x,y,w,h = cv.boundingRect(contours[i])
    img_cut = moban_erzhi[y:y+h,x:x+w]
    resize_cut = cv.resize(img_cut,(18,36))
    list.insert(0,resize_cut)
    print(x,y,w,h)

moban_list = open('D:\pycharm\plate_recognize\moban_list','wb')
pickle.dump(list,moban_list)
moban_list.close()
# moban_file = open('moban_list','rb')
# list = pickle.load(moban_file)

# cv.imshow('1',list[0])
# plt.imshow(moban)
# plt.show()


# pickle_file = open('moban.pkl','wb')


cv.waitKey(0)
cv.destroyAllWindows()
import cv2
import os
import numpy as np

def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
         
        size = len(filename)
        labels.append(filename[0:size-4])
        
        if img is not None:
            images.append(img)
    return images,labels


images = []
labels = []

kernel = np.ones((6,6), np.uint8) 
images , labels = load_images_from_folder("train")


# downstep = 2
# rightstep = 5
# box = 140
# total = box * box

# for i in range(0,150,downstep):
# 	for j in range (0,600,rightstep):
# 		for x in range(box):
# 			for y in range(box):
# 				temp = images[0][i+x][j+y][0]+images[0][i+x][j+y][1]+images[0][i+x][j+y][2]
# 				mapp[temp] = mapp[temp] + 1
# 	for i in mapp :
# 		mapp[i] = mapp[i]/total
# 		if mapp[i] > threshold :




for i in range(len(images)):
    img_gray = cv2.cvtColor(images[i], cv2.COLOR_BGR2HSV)
    images[i] = img_gray

cv2.imshow("dd",images[0])
cv2.waitKey(0)

kernel = np.ones((6,6), np.uint8) 

for i in range(1):
    img=cv2.dilate(images[i],kernel,iterations=1)
    images[i] = img

cv2.imshow("dd",images[0])
cv2.waitKey(0)

# num1 = images[0][0][0][0]
# num2 =images[0][0][0][1]
# num3 =images[0][0][0][2]

# f = open("lol.txt","w+")
# for i in range(150):
# 	for j in range (600):
# 		if images[0][i][j][0] == num1 and images[0][i][j][1]==num2 and images[0][i][j][2]== num3  :
# 			images[0][i][j][0]=255
# 			images[0][i][j][1]=255
# 			images[0][i][j][2]=255
# 		f.write(str(images[0][i][j]))
# 	f.write("\n")	
# f.close() 
# cv2.imshow("dd",images[0])
# cv2.waitKey(0)


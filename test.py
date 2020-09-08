import os
import numpy as np
import cv2
import pickle
import preprocessing


clf = pickle.load(open("OVA_CSVM", "rb"))

true, false = 0, 0
for pics in os.listdir('test_samples'):
    y = cv2.imread(os.path.join('test_samples', pics))

    y = cv2.cvtColor(y, cv2.COLOR_RGB2HSV)
    _, _, y = cv2.split(y)
	
    y = preprocessing.denoise_img(y)
    

    list_char_imgs = []
    list_char_imgs = preprocessing.segment_image(y)

    strc = '' 
    for i in list_char_imgs:
        strc += chr(65+clf.predict(i.reshape(1, 64*64))[0])

    if pics[:len(pics)-4] == strc:
        true += 1
    else:
        
        false += 1



print(true/(true+false))
print(false)

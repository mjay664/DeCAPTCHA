import numpy as np
import pickle
import cv2

# DO NOT CHANGE THE NAME OF THIS METHOD OR ITS INPUT OUTPUT BEHAVIOR

# INPUT CONVENTION
# filenames: a list of strings containing filenames of images

# OUTPUT CONVENTION
# The method must return a numpy array (not numpy matrix or scipy matrix) and a list of strings.
# Make sure that the length of the array and the list is the same as the number of filenames that 
# were given. The evaluation code may give unexpected results if this convention is not followed.
def segment_image(img):
    images = []

    (dim1, dim2) = img.shape
    r = img.sum(axis=0)

    threshold = dim1 * 255
    init_ind = -1
    for i in range(dim2):
        if r[i] < threshold-510:
            if init_ind == -1:
                init_ind = i
        if init_ind != -1 and (r[i] >= threshold-510 or dim2-1==i) and (r[i - 1] < threshold-510 or dim2-1==i):
            tmp = img[:, init_ind:i]

            s = tmp.sum(axis=1)
            th = 255 * (i - init_ind)
            r_i = -1
            for i in range(dim1):
                if s[i] < th-510 and r_i == -1:
                    r_i = i
                if r_i != -1 and s[i] >= th-510 and s[i - 1] < th-510:
                    tmp = tmp[r_i:i, :]
                    break

            init_ind = -1
            tmp = cv2.resize(tmp, (64, 64))
            images.append(tmp)

    return images


def denoise_img(img, kernel=np.ones((5, 5), np.float), sigma=0.5, erosion_iterations=1):
    v = img

    v = cv2.erode(v, kernel, iterations=erosion_iterations)

    img_gray = cv2.GaussianBlur(v,(5,5), sigma, cv2.BORDER_ISOLATED)

    _, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

    thresh = cv2.dilate(thresh, kernel, iterations=erosion_iterations)

    return thresh


def decaptcha( filenames ):
    numChars = 3 * np.ones( (len( filenames ),) )
    # The use of a model file is just for sake of illustration
    clf = pickle.load(open("OVA_CSVM", "rb"))
    codes = []

    for i in range(len(filenames)):
        
        im = cv2.imread(filenames[i])
        
        im = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
        _, _, im = cv2.split(im)

        im = denoise_img(im)
            
        list_im = segment_image(im)

        numChars[i] = len(list_im)

        strc = ''
        for j in list_im:
            strc += chr(65+clf.predict(j.reshape(1, 64*64))[0])

        codes.append(strc)   
    
    return (numChars, codes)

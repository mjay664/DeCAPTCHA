import cv2
import os
import numpy as np
from scipy import ndimage


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


def show_image(img, str=mat):
    cv2.imshow(str, img)
    cv2.waitKey()


def denoise_img(img, kernel=np.ones((5, 5), np.float), sigma=0.5, erosion_iterations=1):
    v = img

    v = cv2.erode(v, kernel, iterations=erosion_iterations)
    show_image(v, "Erosion")

    img_gray = cv2.GaussianBlur(v,(5,5), sigma, cv2.BORDER_ISOLATED)
    show_image(img_gray, "Blur")

    _, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    show_image(thresh, "Binary")

    thresh = cv2.dilate(thresh, kernel, iterations=erosion_iterations)
    show_image(thresh, "Dialate")

    return thresh


def load_images_from_folder(folder):
    kernel = np.ones((5, 5), np.float)

    images = []
    labels = []
    threshes = []
    r = 0
    for filename in os.listdir(folder):
        r = r + 1
        if r > 2:
            exit(0)
        img = cv2.imread(os.path.join(folder, filename))
        show_image(img, "org")

        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        show_image(img, "HSV")

        h, s, v = cv2.split(img)
        show_image(img)

        thresh = denoise_img(v, kernel, 0.5, 1)

        size = len(filename)
        labels.append(filename[0:size - 4])

        xxc = segment_image(thresh)
        cc = -1
        for i in xxc:
            # print(i.shape)
            show_image(i)
            cc = cc + 1
            cv2.imwrite("featureimg/" + filename[cc] + str(r) + str(cc) + ".png", i)

        if img is not None:
            threshes.append(thresh)
            images.append(img)
    return images, labels, threshes


# main fn##################################################################
if __name__ == '__main__':
    images = []
    labels = []

    images, labels, threshes = load_images_from_folder("train")

    count = 0
    for filename in os.listdir("train"):
        count += len(filename) - 4
    print(count)

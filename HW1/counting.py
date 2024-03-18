import os.path
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Set paths
# Due to the wrong naming of the Aluminum image, I have already renamed it from AI.jpg to Al.jpg
al_path = os.path.join('image', 'Al.jpg')
fe_path = os.path.join('image', 'Fe.jpg')
p_path = os.path.join('image', 'P.jpg')

# Read images
al = cv2.imread(al_path)
fe = cv2.imread(fe_path)
p = cv2.imread(p_path)


# RGB TO HSV
def rgb2hsv(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # split H, S, V
    h, s, v = cv2.split(hsv_img)
    return [h, s, v]


# Plot image
def show_image3(imgs: list):
    plt.figure(figsize=(15, 15))
    for i, img in enumerate(imgs):
        plt.subplot(1, 3, i + 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
    plt.show()


# Plot HSV image
def show_hsv_image(imgs: list):
    cv2.imshow('H', imgs[0])
    cv2.imshow('S', imgs[1])
    cv2.imshow('V', imgs[2])
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Remove black and white points and logo
def remove_light(split_img: list):
    h, s, v = split_img
    for i in range(h.shape[0]):
        for j in range(h.shape[1]):
            # Remove black points
            if 0 <= h[i, j] <= 180 and 0 <= v[i, j] <= 46:
                h[i, j] = 0
                s[i, j] = 0
                v[i, j] = 0
            # Remove white points
            elif 0 <= h[i, j] <= 180 and 221 <= v[i, j] <= 255 and 0 <= s[i, j] <= 30:
                h[i, j] = 0
                s[i, j] = 0
                v[i, j] = 0
            # Remove logo area
            if i <= 70 and 180 <= v[i, j] <= 255:
                h[i, j] = 0
                s[i, j] = 0
                v[i, j] = 0

    return [h, s, v]


# Counting
def count_points(img: list):
    h, s, v = img
    counter = 0
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            if v[i, j] != 0:
                counter += 1

    return counter


images = [al, fe, p]
# show_image3(images)

hsv_al = rgb2hsv(al)
hsv_fe = rgb2hsv(fe)
hsv_p = rgb2hsv(p)
# show_hsv_image(hsv_al)

# pre-process
processed_al = remove_light(hsv_al)
processed_fe = remove_light(hsv_fe)
processed_p = remove_light(hsv_p)

# counting
counter_al = count_points(processed_al)
counter_fe = count_points(processed_fe)
counter_p = count_points(processed_p)

print("Al元素的点数：", counter_al)
print("Fe元素的点数：", counter_fe)
print("P元素的点数：", counter_p)


# 求解重叠量，并将两者的重叠量画出来
def count2overlap(img1: list, img2: list):
    h1, s1, v1 = img1
    h2, s2, v2 = img2
    h0 = np.zeros(h1.shape, dtype=np.uint8)
    s0 = np.zeros(s1.shape, dtype=np.uint8)
    v0 = np.zeros(v1.shape, dtype=np.uint8)
    counter = 0
    for i in range(v1.shape[0]):
        for j in range(v1.shape[1]):
            if v1[i, j] != 0 and v2[i, j] != 0:
                h0[i, j] = h1[i, j] + h2[i, j]
                s0[i, j] = s1[i, j] + s2[i, j]
                v0[i, j] = v1[i, j] + v2[i, j]
                counter += 1

    return counter, [h0, s0, v0]


# 求解三者的重叠量
def count3overlap(img1: list, img2: list, img3: list):
    h1, s1, v1 = img1
    h2, s2, v2 = img2
    h3, s3, v3 = img3
    h0 = np.zeros(h1.shape, dtype=np.uint8)
    s0 = np.zeros(s1.shape, dtype=np.uint8)
    v0 = np.zeros(v1.shape, dtype=np.uint8)
    counter = 0
    for i in range(v1.shape[0]):
        for j in range(v1.shape[1]):
            if v1[i, j] != 0 and v2[i, j] != 0 and v3[i, j] != 0:
                h0[i, j] = h1[i, j] + h2[i, j] + h3[i, j]
                s0[i, j] = s1[i, j] + s2[i, j] + s3[i, j]
                v0[i, j] = v1[i, j] + v2[i, j] + v3[i, j]
                counter += 1

    return counter, [h0, s0, v0]


# Get 2 overlap
overlap_al_fe, overlap_img_af = count2overlap(processed_al, processed_fe)
overlap_al_p, overlap_img_ap = count2overlap(processed_al, processed_p)
overlap_fe_p, overlap_img_fp = count2overlap(processed_fe, processed_p)
overlap_img_af = cv2.cvtColor(cv2.merge(overlap_img_af), cv2.COLOR_HSV2BGR)
overlap_img_ap = cv2.cvtColor(cv2.merge(overlap_img_ap), cv2.COLOR_HSV2BGR)
overlap_img_fp = cv2.cvtColor(cv2.merge(overlap_img_fp), cv2.COLOR_HSV2BGR)
print("Al和Fe的重叠量：", overlap_al_fe)
print("Al和P的重叠量：", overlap_al_p)
print("Fe和P的重叠量：", overlap_fe_p)
# show_image3([overlap_img_af, overlap_img_ap, overlap_img_fp])

# Get 3 overlap
overlap_all, overlap_img_all = count3overlap(processed_al, processed_fe, processed_p)
overlap_img_all = cv2.cvtColor(cv2.merge(overlap_img_all), cv2.COLOR_HSV2BGR)
print("Al, Fe和P的重叠量：", overlap_all)
# cv2.imshow('Overlay with Al, Fe and P elements', overlap_img_all)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Save images
if not os.path.exists('image_out'):
    os.makedirs('image_out')
cv2.imwrite(os.path.join('image_out', 'overlap_img_af.jpg'), overlap_img_af)
cv2.imwrite(os.path.join('image_out', 'overlap_img_ap.jpg'), overlap_img_ap)
cv2.imwrite(os.path.join('image_out', 'overlap_img_fp.jpg'), overlap_img_fp)
cv2.imwrite(os.path.join('image_out', 'overlap_img_all.jpg'), overlap_img_all)

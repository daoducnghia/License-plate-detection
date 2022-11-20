import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
from PIL import Image

img_ori = cv2.imread('input/demo.jpg')
#print("Kich co anh:", img_ori.shape)

#Chuyển thành ảnh xám
gray = cv2.cvtColor(img_ori,cv2.COLOR_BGR2GRAY)

#Cân bằng lược đồ xám
equal_histogram = cv2.equalizeHist(gray)
# cv2.imshow("Anh xam", gray)
# cv2.waitKey()

#Tách ngưỡng
threshold = cv2.adaptiveThreshold(equal_histogram,255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2)

#Lấy biên có 4 cạnh
contours, h = cv2.findContours(threshold, 1, 2)

largest_rectangle = [0,0]
for c in contours:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.01 * peri, True)
    if len(approx) == 4:
        area = cv2.contourArea(c)
        if area > largest_rectangle[0]:
            largest_rectangle = [cv2.contourArea(c),c,approx]

x,y,w,h = cv2.boundingRect(largest_rectangle[1])
#Vẽ khung
cv2.drawContours(img_ori, [largest_rectangle[1]], 0, (0,255,0), 8)
#Hiển thị khung ảnh
# cv2.imshow("Danh dau doi tuong", img_ori)
# cv2.waitKey()

#Cắt khung biển số xe
crop = img_ori[y:y+h, x:x+w]
#Hiển thị ảnh khung biển số xe
cv2.imshow("Khung bien so", crop)
cv2.waitKey()

pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

#Chuyển ảnh xám
gray = cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
#Lọc nhiễu bằng GaussianBlur
blur = cv2.GaussianBlur(gray, (3,3), 0)
#Tách ngưỡng
thresh = cv2.threshold(blur,0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN, kernel, iterations = 1)

invert = 255 - opening
#Chuyển các ký tự trên ảnh thành chữ
data = pytesseract.image_to_string(invert, lang='eng', config='--psm 6')

print("Bien so xe:", data)
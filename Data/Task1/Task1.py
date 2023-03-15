import cv2
import pytesseract
import os

def Img_to_string(image_path):
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' #Path to the tesseract .exe 
    img = cv2.imread(image_path) #Path to image
    print(pytesseract.image_to_string(img)) 

image_path = 'C:\My Files/Programming/Project/Data/Task1/1.jpg'
Img_to_string(image_path)


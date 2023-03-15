import easyocr
import numpy as np
from matplotlib import pyplot as plt
import cv2



def img_to_text(img_path):
    '''
    I started by creating new environment 
    than i installed pytorch, opencv, easyocr, matplotlib and etc. so that i can use it
    this time i wanted to have different approach than the last time i did.
    '''
    reader  = easyocr.Reader(['en'], gpu=False)
    result = reader.readtext(img_path)
    # print(result) if i print this i will get the eastimated text placements and i would use this to plot them
    top_left = tuple(result[0][0][0])
    bottom_right = tuple(result[3][0][2])
    text = result[0][1]
    font = cv2.FONT_HERSHEY_SIMPLEX
    # plotting the image
    img = cv2.imread(img_path)
    img = cv2.rectangle(img, top_left, bottom_right, (0,255,0),5)
    img = cv2.putText(img, text, top_left, font, 2, (255,255,255),2,cv2.LINE_AA)
    plt.imshow(img)
    plt.show()
# Now we can just give path and we will have the annotated image as an output
img_path = "C:\My Files/Programming/Project/Data/task3/freezer_image (1).jpg"
img_to_text(img_path)
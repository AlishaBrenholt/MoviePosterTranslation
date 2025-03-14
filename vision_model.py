import torch
# Read Poster data

# Test data reader
data_path = "./data/images/"
poster_name = "17/en/mickey1_en.jpg"

# More Generalized data reader (iterative)
# import os
# from os import walk
# for (dirpath, dirnames, filenames) in walk(data_path):
#     print(f"Directory_path : {dirpath}")
#     print(f"Folder name : {dirnames}")
#     # print(f"File names : {filenames}")

import cv2
import pytesseract

# create tesseract instance
# conda install pytesseract
# You can find the path using 'which tesseract' command in terminal in macOS
tesseract_path = "/Users/kiyujin/miniconda3/envs/pytorch/bin/tesseract"
pytesseract.pytesseract.tesseract_cmd = tesseract_path
print(pytesseract.get_tesseract_version())
# Load image
poster = cv2.imread(data_path + poster_name)
# gray conversion improves accuracy
gray_poster = cv2.cvtColor(poster, cv2.COLOR_BGR2GRAY)

# Display the poster
# cv2.imshow("gray", gray_poster)
# cv2.waitKey(0)

custom_ocv_config = r'--oem 3 --psm 6'
data = pytesseract.image_to_data(gray_poster, config=custom_ocv_config, output_type=pytesseract.Output.DICT)

# confidence threshold
# confidence_threshold = 50

# Iterate through detected text
text_list = []
for i in range(len(data['text'])):
    text = data['text'][i].strip()
    confidence = int(data['conf'][i])
    # if text and confidence > confidence_threshold:
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]

        # Draw bounding box for the text
        cv2.rectangle(poster, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Put the text above the box
        cv2.putText(poster, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

cv2.imshow("poster", poster)
cv2.waitKey(0)
cv2.destroyAllWindows()
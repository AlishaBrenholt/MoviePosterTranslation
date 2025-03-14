import torch
# Read Poster data

# Test data reader
data_path = "./data/images/17/en/"
poster_name = "mickey1_en.jpg"

# More Generalized data reader (iterative)
# import os
# from os import walk
# for (dirpath, dirnames, filenames) in walk(data_path):
#     print(f"Directory_path : {dirpath}")
#     print(f"Folder name : {dirnames}")
#     # print(f"File names : {filenames}")

import cv2
import pytesseract
import easyocr
import os

# create tesseract instance
# conda install pytesseract
# You can find the path using 'which tesseract' command in terminal in macOS
tesseract_path = "/Users/kiyujin/miniconda3/envs/pytorch/bin/tesseract"
pytesseract.pytesseract.tesseract_cmd = tesseract_path
# Load image
poster = cv2.imread(data_path + poster_name)

# print(results)
# Display the poster
# cv2.imshow("gray", gray_poster)
# cv2.waitKey(0)

# Never mind easyocr sucks
def easyocr_extractor(poster):
    # Convert to grayscale (improves text contrast)
    gray = cv2.cvtColor(poster, cv2.COLOR_BGR2GRAY)

    # Apply thresholding (better text clarity)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Save & use the preprocessed image
    cv2.imwrite("processed.jpg", thresh)  # Save processed image

    reader = easyocr.Reader(['en'])
    results = reader.detect(poster)
    print(len(results))
    print(results[1])
    print(len(results[0][0]))
    print(results[0][0])
    for (box, text) in results:

        (top_left, top_right, bottom_right, bottom_left) = box
        top_left = tuple(map(int, top_left))
        bottom_right = tuple(map(int, bottom_right))

        cv2.rectangle(poster, top_left, bottom_right, (0, 255, 0), 2)
        text_loc = max(top_left[1] - 10, 10)
        cv2.putText(poster, text, (top_left[0], text_loc), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("poster", poster)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# easyocr_extractor(poster)



def tesseract_extractor(poster):
    # gray conversion improves accuracy
    gray_poster = cv2.cvtColor(poster, cv2.COLOR_BGR2GRAY)

    # _, thresh = cv2.threshold(gray_poster, 150, 255, cv2.THRESH_BINARY)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    # processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # gray_poster = cv2.adaptiveThreshold(processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    custom_ocv_config = r'--oem 3 --psm 6'
    data = pytesseract.image_to_data(gray_poster, config=custom_ocv_config, output_type=pytesseract.Output.DICT)


    # confidence threshold
    confidence_threshold = 30

    # Iterate through detected text
    text_list = []
    for i in range(len(data['text'])):
        text = data['text'][i].strip()
        confidence = int(data['conf'][i])
        # if text:
        if text and confidence > confidence_threshold:
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]

            # Draw bounding box for the text
            cv2.rectangle(poster, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # Put the text above the box
            cv2.putText(poster, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow("poster", poster)
    cv2.waitKey(0)
    # result_path = 'results/' + 'pre' + poster_name[:-4]+ custom_ocv_config + " conf" + str(confidence_threshold) +'.jpg'
    result_path = 'results/' + poster_name[:-4] + custom_ocv_config + " conf" + str(confidence_threshold) +'.jpg'
    print(result_path)
    cv2.imwrite(result_path, poster)

    cv2.destroyAllWindows()

tesseract_extractor(poster)

# Tried contour. Maybe can add this later.
def test_tess_extractor(poster):

    # Convert to grayscale
    gray = cv2.cvtColor(poster, cv2.COLOR_BGR2GRAY)

    # Apply binary thresholding to highlight text
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours in the thresholded image (which may correspond to text regions)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop through contours to filter out non-text regions based on area and shape
    for contour in contours:
        # Filter small contours, since non-text regions usually have small areas
        area = cv2.contourArea(contour)

        area_th = 500000
        if area > area_th:  # You can adjust this area threshold to fit your image
            # Get bounding box for each contour
            x, y, w, h = cv2.boundingRect(contour)

            # Draw bounding box around text-like regions
            cv2.rectangle(poster, (x, y), (x + w, y + h), (0, 255, 0), 2)

    result_path = 'results/' + poster_name[:-4]  + " contour" +  str(area_th) +'.jpg'
    print(result_path)
    cv2.imwrite(result_path, poster)

    cv2.destroyAllWindows()

# test_tess_extractor(poster)
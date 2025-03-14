import torch
# Read Poster data

# Test data reader
data_path = "./data/images/207/en/"
poster_name = "1xYoAVb0zOsLZd39SIQltn5r6JE.jpg"
# poster_name = "2eWjdb9WcKRUPFaUetfHNDqyyWA.jpg"
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

# Function to sample 3 pixels around the bounding box to get the background color
def get_background_color(image, x, y, w, h):
    # Sample from 3 pixels next to the text bounding box (e.g., to the right of the bounding box)
    sample_x = x + w + 30  # 3 pixels to the right of the bounding box
    sample_y = y + h + 10  # Take a sample from the middle of the bounding box

    # Ensure we are within bounds of the image
    sample_x = min(sample_x, image.shape[1] - 1)
    sample_y = min(sample_y, image.shape[0] - 1)

    # Get the pixel color from that point
    background_color = image[sample_y, sample_x]
    return tuple(background_color)

def tesseract_extractor(data_path, poster_name, tesseract_path):
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
    # Load image
    input_path = data_path + poster_name
    poster = cv2.imread(input_path)

    # gray conversion improves accuracy
    gray_poster = cv2.cvtColor(poster, cv2.COLOR_BGR2GRAY)

    # _, thresh = cv2.threshold(gray_poster, 150, 255, cv2.THRESH_BINARY)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    # processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # gray_poster = cv2.adaptiveThreshold(processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    custom_ocv_config = r'--oem 3 --psm 11'
    data = pytesseract.image_to_data(gray_poster, config=custom_ocv_config, output_type=pytesseract.Output.DICT)


    # confidence threshold
    # confidence_threshold = 1000

    # Iterate through detected text
    text_list = []
    for i in range(len(data['text'])):
        text = data['text'][i].strip()
        confidence = int(data['conf'][i])
        if text:
        # if text and confidence > confidence_threshold:
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]

            # Extract the region with the detected text
            text_region = poster[y:y + h, x:x + w]

            # Get the background color from the neighboring pixels
            background_color = get_background_color(poster, x, y, w, h)

            # Fill the text region with the background color
            poster[y:y + h, x:x + w] = background_color

            # Bounding box drawing section starts
            # Draw bounding box for the text
            # cv2.rectangle(poster, (x, y), (x + w, y + h), (0, 0, 255), 2)
            #
            # # Put the text above the box
            # cv2.putText(poster, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            # Bounding box drawing section ends

    # cv2.imshow("poster", poster)
    # cv2.waitKey(0)
    # bounding box version result save
    # result_path = 'results/text_recognition/' + 'pre' + poster_name[:-4]+ custom_ocv_config + " conf" + str(confidence_threshold) +'.jpg'
    # blurred version result save
    result_path = 'results/blurred/' + poster_name[:-4] + custom_ocv_config +'.jpg'
    cv2.imwrite(result_path, poster)
    cv2.destroyAllWindows()
    return data

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

# EAST : Deep Learning
# def EAST_extractor(poster):
#     import cv2
#
#     # Load the pre-trained EAST text detector
#     net = cv2.dnn.readNet("frozen_east_text_detection.pb")  # Replace with your EAST model path
#
#     # Load the image
#     image_path = "movie_poster.jpg"
#     image = cv2.imread(image_path)
#
#     # Prepare the image for text detection
#     height, width = image.shape[:2]
#     blob = cv2.dnn.blobFromImage(image, 1.0, (width, height), (123.68, 116.78, 103.94), swapRB=True, crop=False)
#     net.setInput(blob)
#
#     # Get the output layer
#     scores, geometry = net.forward(["score1", "geometry1"])
#
#     # Post-process the output to extract the text bounding boxes
#     # (This involves thresholding and non-maximum suppression to eliminate redundant boxes)
#
#     # Process the scores and geometry to extract the text boxes
#     # (Refer to the EAST model documentation for detailed implementation)
#
#     # For simplicity, let's just show the image with some bounding boxes
#     for box in boxes:  # Assuming 'boxes' is a list of detected bounding boxes from EAST
#         cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
#
#     # Display the image
#     cv2.imshow("Detected Text", image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

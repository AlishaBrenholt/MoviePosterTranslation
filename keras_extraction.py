import keras_ocr
import matplotlib.pyplot as plt
import cv2
import numpy as np


def remove_text_opencv(data_path, poster_name):
    image = cv2.imread(data_path + poster_name)
    read_image = keras_ocr.tools.read(data_path + poster_name)


    pipeline = keras_ocr.pipeline.Pipeline()

    prediction_groups = pipeline.recognize([read_image])

    # Create a mask for inpainting
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    for _, box in prediction_groups[0]:
        box = np.array(box).astype(int)
        cv2.fillPoly(mask, [box], (255, 255, 255))  # Fill detected text area in mask

    # Apply inpainting to remove text
    blurred_image = cv2.inpaint(image, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

    result_path = './results/blurred/' + data_path[24:-4] + poster_name
    cv2.imwrite(result_path, blurred_image)
    return result_path

def keras_extractor(data_path, poster_name, blur_strength=(190,190)):
    # To check if the image is shown correctly
    # original_image = cv2.imread(data_path+poster_name)
    # original_image = original_image[:, :, ::-1]
    # plt.imshow(original_image)
    # plt.show()
    # Ensure blur kernel size is odd
    blur_strength = tuple((s if s % 2 == 1 else s + 1) for s in blur_strength)

    # Initialize pipeline
    pipeline = keras_ocr.pipeline.Pipeline()
    # Read in image path
    read_image = keras_ocr.tools.read(data_path+poster_name)
    # result is a list of (word, box) tuples
    prediction_groups = pipeline.recognize([read_image])
    # Plot the predictions
    # keras_ocr.tools.drawAnnotations(image=read_image, predictions=prediction_groups[0])

    # Blurring the bounding box
    image = cv2.imread(data_path+poster_name)
    for text, box in prediction_groups[0]:
        box = np.array(box).astype(int)
        x_min, y_min = np.min(box[:, 0]), np.min(box[:, 1])
        x_max, y_max = np.max(box[:, 0]), np.max(box[:, 1])
        # Blur each character
        text_width = x_max - x_min
        text_height = y_max - y_min
        char_cnt = len(text)

        if char_cnt > 0:
            char_width = text_width / char_cnt

            for i in range(char_cnt):
                char_x_min = int(x_min + i * char_width)
                char_x_max = int(x_min + (i + 1) * char_width)

                if char_x_max > x_max:
                    char_x_max = x_max

                above_y_min = max(0, y_min - text_height // 2)
                above_y_max = y_min
                noise = 20
                background_color = np.median(image[above_y_min-noise:above_y_max+noise, char_x_min-noise:char_x_max+noise], axis=(0, 1))

                # Extract the character region
                char_region = image[y_min-noise:y_max+noise, char_x_min-noise:char_x_max+noise]

                # Apply blur
                blurred_char = cv2.GaussianBlur(char_region, blur_strength, 0)

                # Blend with extracted color
                blended_char = cv2.addWeighted(blurred_char, 0.7,
                                               np.full_like(blurred_char, background_color, dtype=np.uint8), 0.3, 0)

                # Replace the original character with the blended result
                image[y_min-noise:y_max+noise, char_x_min-noise:char_x_max+noise] = blended_char


    result_path = './results/blurred/' + data_path[24:-4] + "_" + poster_name

    cv2.imwrite(result_path, image)
    return prediction_groups
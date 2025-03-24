import keras_ocr
import matplotlib.pyplot as plt
import cv2

def keras_extractor(data_path, poster_name):
    # To check if the image is shown correctly
    # original_image = cv2.imread(data_path+poster_name)
    # original_image = original_image[:, :, ::-1]
    # plt.imshow(original_image)
    # plt.show()

    # Initialize pipeline
    pipeline = keras_ocr.pipeline.Pipeline()
    # Read in image path
    read_image = keras_ocr.tools.read(data_path+poster_name)
    # result is a list of (word, box) tuples
    prediction_groups = pipeline.recognize([read_image])
    # Plot the predictions
    keras_ocr.tools.drawAnnotations(image=read_image, predictions=prediction_groups[0])

    plt.show()
    return prediction_groups
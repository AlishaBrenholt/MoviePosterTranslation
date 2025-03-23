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

    return prediction_groups



# import keras_ocr
#
# # keras-ocr will automatically download pretrained
# # weights for the detector and recognizer.
# pipeline = keras_ocr.pipeline.Pipeline()
#
# # Get a set of three example images
# images = [
#     keras_ocr.tools.read(url) for url in [
#         'https://upload.wikimedia.org/wikipedia/commons/b/bd/Army_Reserves_Recruitment_Banner_MOD_45156284.jpg',
#         'https://upload.wikimedia.org/wikipedia/commons/e/e8/FseeG2QeLXo.jpg',
#         'https://upload.wikimedia.org/wikipedia/commons/b/b4/EUBanana-500x112.jpg'
#     ]
# ]
#
# # Each list of predictions in prediction_groups is a list of
# # (word, box) tuples.
# prediction_groups = pipeline.recognize(images)
#
# # Plot the predictions
# fig, axs = plt.subplots(nrows=len(images), figsize=(20, 20))
# for ax, image, predictions in zip(axs, images, prediction_groups):
#     keras_ocr.tools.drawAnnotations(image=image, predictions=predictions, ax=ax)

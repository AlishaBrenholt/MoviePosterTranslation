import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
from skimage.metrics import structural_similarity as ssim
import cv2

def check_similarity(image_path, ground_truth_image_path):
    vgg_sim = round(vgg_similarity(image_path, ground_truth_image_path), 3)
    resnet_sim = round(resnet_similarity(image_path, ground_truth_image_path)[0][0], 3)
    struct_sim = round(calculate_ssim(image_path, ground_truth_image_path), 3)
    return [vgg_sim, resnet_sim, struct_sim]

# cosine similarity from the VGG16 model
def vgg_similarity(image_path1, image_path2):
    # Load pre-trained VGG16 model
    base_model = VGG16(weights='imagenet', include_top=False)
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)
    # Preprocess the two images
    img1 = preprocess_image(image_path1)
    img2 = preprocess_image(image_path2)

    # Extract features
    features_img1 = model.predict(img1)
    features_img2 = model.predict(img2)

    # Flatten features and calculate cosine similarity
    features_img1 = features_img1.flatten()
    features_img2 = features_img2.flatten()
    similarity = 1 - cosine(features_img1, features_img2)

    return similarity

# cosine similarity from the ResNet50 model
def resnet_similarity(image_path1, image_path2):
    # Load the pre-trained ResNet50 model
    base_model = ResNet50(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
    img1 = preprocess_image(image_path1)
    img2 = preprocess_image(image_path2)
    img1_features = model.predict(img1).flatten()
    img2_features = model.predict(img2).flatten()
    similarity = cosine_similarity([img1_features], [img2_features])
    return similarity

def calculate_ssim(image_path1, image_path2):
    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)
    dim = (2000, 2000)
    img1 = cv2.resize(img1, dim, interpolation=cv2.INTER_AREA)
    img2 = cv2.resize(img2, dim, interpolation = cv2.INTER_AREA)
    return ssim(img1, img2, channel_axis=2)

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    return img_data
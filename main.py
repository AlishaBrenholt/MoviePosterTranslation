import vision_model
from LLM import LLMController
import poster_generator
from dotenv import load_dotenv
import os
import keras_extraction as ke
from evaluate import EvaluateText
import final_evaluation
import csv

def get_tesseract_path():
    try:
        tesseract_path = os.getenv("TESSERACT_PATH")
    except:
        tesseract_path = "/usr/local/bin/tesseract"
    return tesseract_path

# create tesseract instance
# conda install pytesseract
# You can find the path using 'which tesseract' command in terminal in macOS
load_dotenv()
try:
    tesseract_path = os.getenv("TESSERACT_PATH")
except:
    tesseract_path = "/usr/local/bin/tesseract"

# TODO : delete this codeblock I think? Runs iteratively without it
# data_path = "./data/images/0GOODDATA/parasite/en/"
# poster_name = "en_1.jpg"
'''keras extraction experiment lab'''
# data = ke.keras_extractor(data_path, poster_name)
# print(f"keras_data: {data}")


'''lab done'''

def get_text_from_poster(tesseract_path):
    """
    Extracts text from posters using tesseract. It also saves the images in the results/blurred directory.
    Structure of data_dict:
    movie {
        poster {
                {
                    'level': [], 'page_num' : [], 'block_num' : [], 'line_num' : [], 'word_num' : [],
                    'left' : [], 'top' : [], 'width' : [], 'height' : [], 'conf' : [], 'text' : []
                }
            }
        }

    :param data_path:
    :param tesseract_path:
    :return:
    """
    data_dict = {}
    for movie_folder in os.listdir(movie_path):
        # .DS_Store is a macOS system file, ignore it
        if movie_folder == ".DS_Store":
            continue
        data_dict[movie_folder] = {}
        movie_folder_path = movie_path + movie_folder + "/"
        eng_movie_folder_path = movie_folder_path + "en/"
        kor_movie_folder_path = movie_folder_path + "ko/"
        for poster in os.listdir(eng_movie_folder_path):
            # ensure it ends with .jpg, ko has some .txt files
            if poster.endswith(".jpg"):
                # data_dict[movie_folder][poster] = vision_model.tesseract_extractor(eng_movie_folder_path, poster, tesseract_path)
                data_dict[movie_folder][poster] = ke.remove_text_opencv(eng_movie_folder_path, poster)
    return data_dict

def get_translations(data, llm):
    translation_dict = {}
    counter = 0
    for movie, images in data.items():
        print(f"Translating Movies: {counter}/{len(data)} - {movie}")
        counter +=1
        translation_dict[movie] = {}
        for image, results in images.items():
            word_coords = []
            for pair in results[0]:
                word = pair[0]
                cords = pair[1]
                kor_word = llm.translate_good(word)
                word_coords.append((kor_word, cords))
            translation_dict[movie][image] = word_coords
    return translation_dict

# create tesseract instance
# conda install pytesseract
# You can find the path using 'which tesseract' command in terminal in macOS
load_dotenv()
tesseract_path = get_tesseract_path()
llm = LLMController()
movie_path = "data/images/0GOODDATA/"
data = get_text_from_poster(tesseract_path)
translated_text = get_translations(data, llm)
print(f"Translated Text: {translated_text}")
# evaluator = EvaluateText(llm)
# results = evaluator.evaluate(translated_text)
# print(f"Results: {results}")

# build the english, translated_text pair
BLURRED_PATH = "results/blurred/"
for movie, images in translated_text.items():
    for image, text in images.items():
        image_path = BLURRED_PATH + movie + "_" + image
        # created translated poster
        # stores in results/final_image after running
        text_data = data[movie][image]
        poster_generator.generate_poster(image_path, text, text_data)

# evaluate final poster output
# gathers image similarity for generated poster and ground truth poster
# puts into results/similarity_results.csv
final_images_path = 'results/final_image/'
with (open('results/similarity_results.csv', mode='w', newline='') as csv_file):
    writer = csv.writer(csv_file)
    writer.writerow(["Final Poster Name", "Ground Truth Poster", "VGG16 Similarity",
                     "ResNet50 Similarity", "SSIM"])
    for poster in os.listdir(final_images_path):
        try:
            movie = poster[:-5]
            cur_poster_num = poster[-5]
            ground_truth_poster = movie_path + movie + '/' + 'ko/ko_' + cur_poster_num + '/ko_' + cur_poster_num + '.jpg'
            similarities = final_evaluation.check_similarity(final_images_path+poster, ground_truth_poster)
            writer.writerow([poster, ground_truth_poster] + similarities)
        except:
            print('Filename error: ', ground_truth_poster)
            continue
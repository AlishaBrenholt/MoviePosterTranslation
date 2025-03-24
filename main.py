import vision_model
from LLM import LLMController
import poster_generator
from dotenv import load_dotenv
import os
import keras_extraction as ke

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

# TODO : Make it iterative for later
data_path = "./data/images/apocalypsenow/en/"
poster_name = "en_1.jpg"
'''keras extraction experiment lab'''
data = ke.keras_extractor(data_path, poster_name)
print(f"keras_data: {data}")


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
        data_dict[movie_folder] = {}
        movie_folder_path = movie_path + movie_folder + "/"
        eng_movie_folder_path = movie_folder_path + "en/"
        kor_movie_folder_path = movie_folder_path + "ko/"
        for poster in os.listdir(eng_movie_folder_path):
            # ensure it ends with .jpg, ko has some .txt files
            if poster.endswith(".jpg"):
                # data_dict[movie_folder][poster] = vision_model.tesseract_extractor(eng_movie_folder_path, poster, tesseract_path)
                data_dict[movie_folder][poster] = ke.keras_extractor(eng_movie_folder_path, poster)
    return data_dict

def get_translations(data):
    llm = LLMController()
    translation_dict = {}
    counter = 0
    for movie, images in data.items():
        print(f"Translating Movies: {counter}/{len(data)} - {movie}")
        counter +=1
        translation_dict[movie] = {}
        for image, results in images.items():
            translated_text = []
            for pair in results:
                pair = pair[0]
                translated_text.append(llm.translate_good(pair[0]))
            translation_dict[movie][image] = translated_text
        print(f"Translated dictionary: {translation_dict}")
        return translation_dict # TODO: Delete this line, it's just for testing and making it quicker
    return translation_dict

# create tesseract instance
# conda install pytesseract
# You can find the path using 'which tesseract' command in terminal in macOS
load_dotenv()
tesseract_path = get_tesseract_path()

movie_path = "data/images/0GOODDATA/"
data = get_text_from_poster(tesseract_path)
translated_text = get_translations(data)

# build the english, translated_text pair
BLURRED_PATH = "results/blurred/"
for movie, images in translated_text.items():
    for image, text in images.items():
        image_path = BLURRED_PATH + movie + "/" + image
        # For sam's code? TODO still
        # english_text = data[movie][image]['text']
        # data[movie][image]['text'] = [english_text, text]



# pass filepath to edited poster and the data for it
# stores in results/final_image after running
# final_poster = poster_generator.generate_poster("19OEGyBQtG2OFaaCBxPCvjzOw3--oem 3 --psm 11.jpg",
#                                                 data)
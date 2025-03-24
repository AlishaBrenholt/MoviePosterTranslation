import vision_model
from LLM import LLMController
import poster_generator
from dotenv import load_dotenv
import os
import keras_extraction as ke

# create tesseract instance
# conda install pytesseract
# You can find the path using 'which tesseract' command in terminal in macOS
load_dotenv()
try:
    tesseract_path = os.getenv("TESSERACT_PATH")
except:
    tesseract_path = "/usr/local/bin/tesseract"


# Load image
# TODO : Make it iterative for later
data_path = "./data/images/interstellar/en/"
poster_name = "en_1.jpg"
# Extract texts and save text-extracted poster in results/blurred/
'''
Structure of data
{
    'level': [], 'page_num' : [], 'block_num' : [], 'line_num' : [], 'word_num' : [],
    'left' : [], 'top' : [], 'width' : [], 'height' : [], 'conf' : [], 'text' : [] 
}

data['text'] has the text contents that can go into LLM
'''
# data = vision_model.tesseract_extractor(data_path, poster_name, tesseract_path) # This has text information including text contents and their location

'''keras extraction experiment lab'''
data = ke.keras_extractor(data_path, poster_name)
print(f"keras_data: {data}")
'''lab done'''

# Input data['text'] to LLM part
# llm = LLMController()
# translated_text = []
# for text in data['text']:
#     if text == '':
#         translated_text.append('')
#     else:
#         translated_text.append(llm.translate_good(text))
# # translated_text : translated version of data['text']
# # ex. llm_output = ['', '', '', '', '죽은', '시인들']
# # Put the output into data
# print(translated_text)
# eng_data_text = data['text']
# data['text'] = [eng_data_text, translated_text]
# # Input data into AnyText now.
#
# # pass filepath to edited poster and the data for it
# # stores in results/final_image after running
# final_poster = poster_generator.generate_poster("en_1--oem 3 --psm 11.jpg",
#                                                 data)
import vision_model
import llm
import cv2

# create tesseract instance
# conda install pytesseract
# You can find the path using 'which tesseract' command in terminal in macOS
tesseract_path = "/Users/kiyujin/miniconda3/envs/pytorch/bin/tesseract"
# Load image
# TODO : Make it iterative for later
data_path = "./data/images/207/en/"
poster_name = "1xYoAVb0zOsLZd39SIQltn5r6JE.jpg"
input_path = data_path + poster_name
# Extract texts and save text-extracted poster in results/blurred/
'''
Structure of data
{
    'level': [], 'page_num' : [], 'block_num' : [], 'line_num' : [], 'word_num' : [],
    'left' : [], 'top' : [], 'width' : [], 'height' : [], 'conf' : [], 'text' : [] 
}

data['text'] has the text contents that can go into LLM
'''
data = vision_model.tesseract_extractor(input_path, tesseract_path) # This has text information including text contents and their location

print(data['text'])

# Input data['text'] to LLM part
translated_text = []
for text in data['text']:
    if text == '':
        translated_text.append('')
    else:
        translated_text.append(llm.FullLLM().translate(text))
# translated_text : translated version of data['text']
# ex. llm_output = ['', '', '', '', '죽은', '시인들']
# Put the output into data
data['text'] = translated_text
# Input data into AnyText now.
print(data['text'])
# TODO : Sam's part

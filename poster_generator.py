import cv2
from PIL import Image, ImageDraw, ImageFont
def generate_poster(img_path, data):
    img = Image.open('./results/blurred/' + img_path)
    draw = ImageDraw.Draw(img)
    for i in range(len(data['left'])):
        text = data['text'][i]
        if data['width'][i] > 0 and len(text) > 0:
            fontsize = data['width'][i]//len(text)
            if fontsize > 0:
                font = ImageFont.truetype('./fonts/malgun-gothic.ttf', fontsize)
            else:
                font = ImageFont.truetype('./fonts/malgun-gothic.ttf', 10)
            position = (data['left'][i], data['top'][i])
            text_color = (0, 0, 0)
            draw.text(position, text, text_color, font=font,
                      align='center')
    img.save('./results/final_image/' + img_path)
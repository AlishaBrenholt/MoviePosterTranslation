from PIL import Image, ImageDraw, ImageFont
import math

def generate_poster(img_path, ko_text, text_data):
    img = Image.open(img_path)
    draw = ImageDraw.Draw(img)
    for item in ko_text:
        korean_characters = item[0]
        positions = item[1]
        left = positions[0]
        top = positions[1]
        right = positions[2]
        bottom = positions[3]

        middle_pixel = (left[0] + ((right[0] - left[0]) / 2), bottom[1] + ((top[1] - bottom[1])) / 2)
        fontsize = get_font_size(korean_characters, positions, draw, middle_pixel)
        font = ImageFont.truetype('./fonts/malgun-gothic.ttf', fontsize)
        font_color = get_font_color(img_path, positions)
        draw.text(middle_pixel, korean_characters, font_color, font=font, anchor='mm')

    save_path = './results/final_image/' + img_path[16:-9] + img_path[-5:]
    img.save(save_path)

def get_font_color(img_path, positions):
    top = positions[1]
    bottom = positions[3]
    img = Image.open(img_path)
    pixels = img.load()

    # get pixel from middle of bounding box for text data
    middle_x = top[0] - bottom[0]
    middle_y = top[1] - bottom[1]
    background_color = pixels[middle_x, middle_y]

    # Brightness=0.2126R+0.7152G+0.0722B
    brightness = (0.2126 * background_color[0] + 0.7152 * background_color[1]
                  + 0.0722 * background_color[2])
    if brightness > 128: # color is light
        text_color = (0, 0, 0) # black text
    else:
        text_color = (255, 255, 255) # white text
    return text_color

def get_font_size(korean_text, positions, draw, middle_pixel):
    if len(korean_text) == 0: # possible error in text detection or translation
        return ImageFont.truetype('./fonts/malgun-gothic.ttf', 1)

    left = positions[0]
    top = positions[1]
    right = positions[2]
    bottom = positions[3]
    fontsize = math.ceil(abs(left[0]-right[0])) # initialize

    # make textbox smaller until it fits within bounding box
    while fontsize > 1:
        font = ImageFont.truetype('./fonts/malgun-gothic.ttf', fontsize)
        # left, top, right, bottom
        bbox = draw.textbbox(middle_pixel, korean_text, font=font, anchor='mm')
        box = (math.floor(bbox[0]), math.floor(bbox[1]),
               math.floor(bbox[2]), math.floor(bbox[3]))

        # if text bbox within outer bounding box
        if bbox[0] >= left[0] and bbox[2] <= right[0]:
            draw.rectangle(box, fill=None, outline=0, width=2)
            break # textbox within bounds
        else:
            fontsize -= 1
    return fontsize
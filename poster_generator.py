from PIL import Image, ImageDraw, ImageFont

def generate_poster(img_path, ko_text, text_data):
    img = Image.open(img_path)
    draw = ImageDraw.Draw(img)
    for item in ko_text:
        korean_characters = item[0]
        positions = item[1]
        left = positions[0]
        right = positions[1]
    #   fontsize = (right[0]-left[0])//len(ko_text[0])
        fontsize = 30
        font = ImageFont.truetype('./fonts/malgun-gothic.ttf', fontsize)
        text_color = (0, 0, 0)
        draw.text(left, korean_characters, text_color, font=font,
                  align='center')

    save_path = './results/final_image/' + img_path[16:-9] + img_path[-5:]
    img.save(save_path)
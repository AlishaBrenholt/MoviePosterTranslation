from PIL import Image, ImageDraw, ImageFont

def generate_poster(img_path, ko_text, text_data):
    img = Image.open(img_path)
    draw = ImageDraw.Draw(img)
    for item in text_data:
        en_text = item[0][0]
        positions = item[0][1]
        left = (positions[0][0], positions[0][1])
        right = (positions[1][0], positions[1][1])
        fontsize = (right[0]-left[0])//len(ko_text[0])
        font = ImageFont.truetype('./fonts/malgun-gothic.ttf', fontsize)
        left = (positions[0][0], positions[0][1])
        text_color = (0, 0, 0)
        draw.text(left, ko_text[0], text_color, font=font,
                  align='center')

    save_path = './results/final_image/' + img_path[16:-9] + img_path[-5:]
    img.save(save_path)
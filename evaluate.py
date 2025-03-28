from pathlib import Path
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


class EvaluateText():
    def __init__(self, llm, data_path="data/images/0GOODDATA/"):
        self.llm = llm
        self.data_path = data_path
        self.smoother = SmoothingFunction()

    def evaluate(self, data):
        # Process the data into intended format
        results = {}
        for movie, images in data.items():
            move_blues = []
            for image, translated_text in images.items():
                words = ""
                for word_pair in translated_text:
                    # left is word, right is location
                    word = word_pair[0]
                    words += word + " "
                ground_truth = self.get_korean_file(movie, image)
                blue_score = sentence_bleu(ground_truth, words, weights=(1, 0, 0, 0), smoothing_function=self.smoother.method1)
                move_blues.append(blue_score)
            # add check for none:
            if len(move_blues) == 0:
                move_blues.append(0)
            average_bleu = sum(move_blues) / len(move_blues)
            results[movie] = average_bleu
        return results



    def get_korean_file(self, movie, image_name):
        # remove the extension from the image name
        image_name = image_name.split(".")[0]
        image_name = image_name.replace("en", "ko")
        data_path = Path(f"{self.data_path + movie}/ko/{image_name}/{image_name}.txt")

        # not all data is same format, so slight variation check
        if not data_path.is_file():
            image_name = image_name.replace("_", "")
            image_name = image_name.replace("ko", "")
            print(f"File not found: {data_path}, trying alternative path. image_name is set to {image_name}")
            data_path = Path(f"{self.data_path + movie}/ko/ko_{image_name}/ko_{image_name}.txt")
        if not data_path.is_file():
            image_name = image_name.replace("_", "")
            image_name = image_name.replace("ko", "")
            data_path = Path(f"{self.data_path + movie}/ko/ko_{image_name}/{image_name}_ko.txt")
        if not data_path.is_file():
            image_name = image_name.replace("_", "")
            image_name = image_name.replace("ko", "")
            data_path = Path(f"{self.data_path + movie}/ko/{image_name}_ko/{image_name}_ko.txt")

        # Still could not find file, so raise error
        if not data_path.is_file():
            raise FileNotFoundError(f"The file {data_path} does not exist.")

        # Read the file and remove commas
        with open(data_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        lines = [line.replace(",", "") for line in lines]
        return lines

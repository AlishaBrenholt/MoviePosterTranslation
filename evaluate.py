from pathlib import Path
from nltk.translate.bleu_score import sentence_bleu


class EvaluateText():
    def __init__(self, llm, data_path="data/images/0GOODDATA/"):
        self.llm = llm
        self.data_path = data_path
        self.bleu_score = None
        self.rouge_score = None
        self.meteor_score = None

    def evaluate(self, data):
        """
        Evaluate the translations using BLEU, ROUGE, and METEOR scores.
        :param data: Dictionary containing the original and translated texts.
        :return: Dictionary with evaluation scores.
        """
        # Process the data into intended format
        results = {}
        for movie, images in data.items():
            move_blues = []
            for image, translated_text in images.items():
                words = translated_text[0]
                ground_truth = self.get_korean_file(movie, image)
                blue_score = sentence_bleu(ground_truth, words)
                move_blues.append(blue_score)
            # add check for none:
            if len(move_blues) == 0:
                move_blues.append(0)
            average_bleu = sum(move_blues) / len(move_blues)
            results[movie] = average_bleu
        return results



    def get_korean_file(self, movie, image_name):
        """
        Read the file and return the string with the commas removed from the text.
        :param movie:
        :param image_name:
        :return:
        """
        # remove the extension from the image name
        image_name = image_name.split(".")[0]
        image_name = image_name.replace("en", "ko")
        data_path = Path(f"{self.data_path + movie}/ko/{image_name}/{image_name}.txt")
        if not data_path.is_file():
            raise FileNotFoundError(f"The file {data_path} does not exist.")
        with open(data_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        lines = [line.replace(",", "") for line in lines]
        return lines



        # Initialize the evaluation metrics
        # self.bleu_score = self.llm.evaluate(data['original'], data['translated'])
        # self.rouge_score = self.llm.evaluate(data['original'], data['translated'], metric='rouge')
        # self.meteor_score = self.llm.evaluate(data['original'], data['translated'], metric='meteor')
        #
        # return {
        #     'BLEU': self.bleu_score,
        #     'ROUGE': self.rouge_score,
        #     'METEOR': self.meteor_score
        # }
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables from .env file
class GPTTranslator():
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("GPT_KEY")
        self.client = OpenAI(
          api_key=self.api_key,
        )

    def translate(self, text, sentence = ""):
        result = self.client.chat.completions.create(
            model="gpt-4o-mini",
            store=True,
            messages=[
                {"role": "system", "content": f"You are a translator who translates English to Korean. The given english might not be correct, so try to understand the real english before translating. It will be related to movies in some way. Only repond with the korean hangul translation, say nothing else. I only want you to translate the word given, however translate it in context with this being the entire english sentence {sentence}. I will feed you one english word at a time, in the order of this sentence. "},
                {"role": "user", "content": f"The word I want you to translate, in context of the sentence, is {text}"},
            ]
            )
        print(f"GPT-4o-Mini: {result.choices[0].message.content}")
        return result.choices[0].message.content

    def translate_group(self, data):

        sentence = ""
        for pair in data[0]:
            sentence += pair[0] + " "
        sentence = sentence.strip()
        messages = [
            {"role": "system",
             "content": f"You are a translator who translates English to Korean. The given english might not be correct, so try to understand the real english before translating. It will be related to movies in some way. Only repond with the korean hangul translation, say nothing else. I only want you to translate the word given, however translate it in context with this being the entire english sentence {sentence}. I will feed you one english word at a time, in the order of this sentence. "},
        ]
        word_coords = []
        for pair in data[0]:
            word = pair[0]
            cords = pair[1]
            messages.append({"role": "user", "content": f"{word}"})
            result = self.client.chat.completions.create(
                model="gpt-4o-mini",
                store=True,
                messages=messages
            )
            kor_word = result.choices[0].message.content
            messages.append({"role": "assistant", "content": kor_word})
            word_coords.append((kor_word, cords))
        return word_coords
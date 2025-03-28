from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, pipeline, AutoModelForCausalLM, \
    AutoTokenizer
from nltk.translate.bleu_score import sentence_bleu

class LLMController():
    def __init__(self):

        # Model 1
        print("Loading models")
        self.model = MBartForConditionalGeneration.from_pretrained("SnypzZz/Llama2-13b-Language-translate")
        self.model.eval()
        self.tokenizer = MBart50TokenizerFast.from_pretrained("SnypzZz/Llama2-13b-Language-translate", src_lang="en_XX")

        # Model 2
        self.model2 = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        self.model2.eval()
        self.tokenizer2 = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

        # Model 3
        self.ket_translator = pipeline("translation", model="KETI-AIR-Downstream/long-ke-t5-base-translation-aihub-en2ko", device="cpu")

        # Model 4 https://github.com/fe1ixxu/ALMA - Couldn't get this one to work. Trying facebook instruct
        self.model_instruct = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
        self.tokenizer_instruct = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
        self.model_instruct.eval()

        # Model 5
        self.mbart_finetuned = MBartForConditionalGeneration.from_pretrained("yesj1234/mbart-mmt_mid1_en-ko")
        self.tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        self.mbart_finetuned.eval()
        print("Models loaded")

    def translate_good(self, text):
        model_inputs = self.tokenizer(text, return_tensors="pt")
        generated_tokens = self.model.generate(
            **model_inputs,
            forced_bos_token_id=self.tokenizer.lang_code_to_id["ko_KR"]
        )
        return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

    def translate_facebook_mbart(self, text):
        self.tokenizer2.src_lang = "en_XX"
        encoded_text = self.tokenizer2(text, return_tensors="pt")
        generated_tokens = self.model2.generate(**encoded_text, forced_bos_token_id=self.tokenizer2.lang_code_to_id["ko_KR"])
        return self.tokenizer2.batch_decode(generated_tokens, skip_special_tokens=True)[0]

    def translate_keti_air_T5(self, text):
        translated = self.ket_translator(text, max_length=40)
        return translated[0]['translation_text']

    def translate_mbart_finetuned(self, text):
        encoded_text = self.tokenizer(text, return_tensors="pt")
        generated_tokens = self.mbart_finetuned.generate(**encoded_text, forced_bos_token_id=self.tokenizer.lang_code_to_id["ko_KR"])
        return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

    def translate_instruct(self, text):
        # prompt = f"System: Translate the following English into Korean. The movie name might not be spelled correctly, assume the correct english for your translation. Only respond with the korean translation results\nUser: {text}\nAssistant:"

        messages = [
            {"role": "user", "content": "You are a translator who translates English to Korean. The given english might not be correct, so try to understand the real english before translating. It will be related to movies in some way. Only repond with the korean hangul translation, say nothing else."},
            {"role": "assistant",
             "content": "Understood. I will translate the given English text to Korean, considering the context of movies. Please provide the text you want me to translate."},
            {"role": "user", "content": f"Robert Downy Jr. Chris Evans Mark Ruffalo Aengrs Endame."},
            {"role":"assistant","content": "로버트 다우니 주니어 크리스 에반스 마크 러팔로, 어벤져스 엔드게임."},
            {"role": "user", "content": text}
        ]
        encs = self.tokenizer_instruct.apply_chat_template(messages, return_tensors="pt")

        ids = self.model_instruct.generate(encs, max_new_tokens=100, do_sample=True, top_k=20)
        decoded = self.tokenizer_instruct.batch_decode(ids, skip_special_tokens=True)
        print(f"Decoded: {decoded}")
        # Find the last [/INST] token
        last_inst = decoded[0].rfind("[/INST]")
        result = decoded[0][last_inst:]
        result = result.replace("[/INST]", "")

        # To remove the pronunciation the model seems to include
        left_paren = result.find("(")
        if left_paren != -1:
            result = result[:left_paren]
        print(result)

        # second pass asking the model to ensure all characters are korean characters and that it makes sense
        # second_pass_messages = [
        #     {"role": "user", "content": "Your job is to ensure the following text only has korean characters and that it makes sense. Return the corrected Korean: 포�по팝 먹고 앱벨지: 엔드게임 보았습니다"},
        #     {"role": "assistant", "content": "팝콘을 먹으며 어벤져스를 봤어요: 엔드게임"},
        #     {"role": "user", "content": result}]
        #
        # encs = self.tokenizer_instruct.apply_chat_template(second_pass_messages, return_tensors="pt")
        #
        # ids = self.model_instruct.generate(encs, max_new_tokens=100, do_sample=True, top_k=20)
        # decoded = self.tokenizer_instruct.batch_decode(ids, skip_special_tokens=True)
        # print(f"Decoded second pass: {decoded}")
        # # Find the last [/INST] token
        # last_inst = decoded[0].rfind("[/INST]")
        # result = decoded[0][last_inst:]
        # result = result.replace("[/INST]", "")
        #
        # # To remove the pronunciation the model seems to include
        # left_paren = result.find("(")
        # if left_paren != -1:
        #     result = result[:left_paren]
        # print("Second pass results: " +result)

        return result




    def evaluate_function(self, sentence_pairs, translate_function=None):
        """
        This function is assuming an input of sentence pairs, where each pair is (english, target). It uses BLEU scores.
        It also takes the translate function as an arguemnt, defaulting to the good one.
        """
        translated = []
        for pair in sentence_pairs:
            english = pair[0]
            translated.append(translate_function(english))
        bleu_scores = []
        for i in range(len(translated)):
            bleu_scores.append(sentence_bleu(sentence_pairs[i][1], translated[i]))

        average_bleu = sum(bleu_scores) / len(bleu_scores)
        return average_bleu


    def evaluate(self, sentence_pairs, print_bool=False):
        print(f"Starting good evaluation\n")
        good_bleu = self.evaluate_function(sentence_pairs, self.translate_good)
        print(f"Starting facebook mbart evaluation\n")
        facebook_bleu = self.evaluate_function(sentence_pairs, self.translate_facebook_mbart)
        print(f"Starting KETI evaluation\n")
        keti_bleu = self.evaluate_function(sentence_pairs, self.translate_keti_air_T5)
        print(f"Starting mbart evaluation\n")
        mbart_bleu = self.evaluate_function(sentence_pairs, self.translate_mbart_finetuned)
        print(f"Starting Instruct evaluation\n")
        instruct_bleu = self.evaluate_function(sentence_pairs, self.translate_instruct)
        score_dict = {"good": good_bleu, "facebook": facebook_bleu, "keti": keti_bleu, "mbart": mbart_bleu, "instruct": instruct_bleu}
        # Print results as a table
        if print_bool:
            self.pretty_print(score_dict)
        return score_dict

    def pretty_print(self, score_dict):

        for i, score in score_dict.items():
            print(f'{i}'.ljust(10), end="")
        print()
        for i,score in score_dict.items():
            print(f'{score:.5f}'.ljust(10), end="")

if __name__ == "__main__":
    llm = LLMController()
    sentence_pairs = [("Parasite is a good movie", ["기생충은 좋은 영화입니다"]),("my friends and I went to the movie theater", ["친구들과 영화관에 갔어요","친구들과 영화관에 갔을 때","친구들과 영화관에 갔는데"]),("We ate popcorn and saw The Avengers: Endgame", ["팝콘을 먹으며 어벤져스를 봤어요: 엔드게임"])]
    llm.evaluate(sentence_pairs, print_bool=True)




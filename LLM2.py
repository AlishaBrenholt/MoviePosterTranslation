from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from nltk.translate.bleu_score import sentence_bleu
import torch

class LLMController():
    def __init__(self):
        # Model 1
        self.model = MBartForConditionalGeneration.from_pretrained("SnypzZz/Llama2-13b-Language-translate")
        self.tokenizer = MBart50TokenizerFast.from_pretrained("SnypzZz/Llama2-13b-Language-translate", src_lang="en_XX")

        # Model 2
        self.model2 = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        self.tokenizer2 = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

        # Model 3
        self.ket_translator = pipeline("translation", model="KETI-AIR-Downstream/long-ke-t5-base-translation-aihub-en2ko", device="cpu")

        # Model 4 https://github.com/fe1ixxu/ALMA
        GROUP2LANG = {
            1: ["da", "nl", "de", "is", "no", "sv", "af"],
            2: ["ca", "ro", "gl", "it", "pt", "es"],
            3: ["bg", "mk", "sr", "uk", "ru"],
            4: ["id", "ms", "th", "vi", "mg", "fr"],
            5: ["hu", "el", "cs", "pl", "lt", "lv"],
            6: ["ka", "zh", "ja", "ko", "fi", "et"],
            7: ["gu", "hi", "mr", "ne", "ur"],
            8: ["az", "kk", "ky", "tr", "uz", "ar", "he", "fa"],
        }
        LANG2GROUP = {lang: str(group) for group, langs in GROUP2LANG.items() for lang in langs}
        group_id = LANG2GROUP["ko"]

        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.ALMAXModel = AutoModelForCausalLM.from_pretrained("haoranxu/X-ALMA-13B-Pretrain", torch_dtype=torch.float32, device_map=self.device)
        # 6 has korean it looks like
        self.ALMAXTokenizer = AutoTokenizer.from_pretrained(f"haoranxu/X-ALMA-13B-Group{group_id}", padding_side='left')



        # Model 5
        self.mbart_finetuned = MBartForConditionalGeneration.from_pretrained("yesj1234/mbart-mmt_mid1_en-ko")
        self.tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

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

    def translateALMAX(self, text):
        prompt = f"<s>[INST] Translate this from English to Korean:\nEnglish: {text}。\nKorean: [/INST]"
        chat_style_prompt = [{"role": "user", "content": prompt}]
        prompt = self.ALMAXTokenizer.apply_chat_template(chat_style_prompt, tokenize=False, add_generation_prompt=True)
        input_ids = self.ALMAXTokenizer(prompt, return_tensors="pt", padding=True, max_length=40, truncation=True).input_ids.to(self.device)
        self.ALMAXModel.eval()
        with torch.no_grad():
            generated_ids = self.ALMAXModel.generate(input_ids=input_ids, num_beams=5, max_new_tokens=20, do_sample=True, temperature=0.6, top_p=0.9)
        outputs = self.ALMAXTokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        print(outputs)



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


    def evaluate(self, sentence_pairs, print=False):
        good_bleu = self.evaluate_function(sentence_pairs, self.translate_good)
        facebook_bleu = self.evaluate_function(sentence_pairs, self.translate_facebook_mbart)
        keti_bleu = self.evaluate_function(sentence_pairs, self.translate_keti_air_T5)
        mbart_bleu = self.evaluate_function(sentence_pairs, self.translate_mbart_finetuned)
        score_dict = {"good": good_bleu, "facebook": facebook_bleu, "keti": keti_bleu, "mbart": mbart_bleu}
        # Print results as a table
        if print:
            self.pretty_print(score_dict)
        return good_bleu, facebook_bleu

    def pretty_print(self, score_dict):

        for i, score in score_dict.items():
            print(f'{i}'.ljust(10), end="")
        print()
        for i,score in score_dict.items():
            print(f'{score:.5f}'.ljust(10), end="")


llm = LLMController()
sentence_pairs = [("I studied at home", ["저는 집에서 공부했습니다"]),("you are very smart", ["당신은 매우 똑똑합니다","너는 매우 똑똑하다","당신은 너무 똑똑합니다"]),("I am a student", ["저는 학생입니다","저는 학생이애요"])]
llm.translateALMAX("I studied at home")
# llm.evaluate(sentence_pairs, print=True)




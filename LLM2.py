from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, pipeline
class LLMController():
    def __init__(self):
        self.model = MBartForConditionalGeneration.from_pretrained("SnypzZz/Llama2-13b-Language-translate")
        self.tokenizer = MBart50TokenizerFast.from_pretrained("SnypzZz/Llama2-13b-Language-translate", src_lang="en_XX")

    def translate(self, text):
        model_inputs = self.tokenizer(text, return_tensors="pt")
        generated_tokens = self.model.generate(
            **model_inputs,
            forced_bos_token_id=self.tokenizer.lang_code_to_id["ko_KR"]
        )
        return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)


# model = "KRAFTON/KORani-v3-13B"
# translator = pipeline(
#     "translation",
#     model=model,
#     device="cpu"  # force usage of CPU
# )
#
# text_to_translate = "Once upon a time"
# translations = translator(text_to_translate)
# print(translations)



# translator = pipeline(
#     "translation",
#     model="Helsinki-NLP/opus-mt-tc-big-en-ko",
#     device="cpu"  # force usage of CPU
# )
#
# text_to_translate = "Once upon a time"
# translations = translator(text_to_translate)
# print(translations)
from google_trans_new import google_translator
import googletrans
# print(googletrans.LANGUAGES)
translator = google_translator()
print(translator.translate('how are you?',lang_src='en', lang_tgt='gu'))
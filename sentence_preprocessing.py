import re
from unidecode import unidecode

def split_sentences(text):
    return re.split('[\n]{1,}|\.\W|\.(?=[A-Z])', text)

def clean_sentence(text):
    return unidecode(re.sub('[^a-zA-Z]', ' ', text).lower().strip())

def _is_meaningfull_sentence(sentence, min_characters, min_fraction_letters):
    number_of_characters = len(sentence)
    number_of_letters = len(re.sub('[^a-zA-Z]', '', sentence))
    fraction_of_letters = number_of_letters / (number_of_characters + 1)
    return number_of_characters > min_characters and fraction_of_letters > min_fraction_letters
    

def filter_sentences(sentences, max_sentences=None, min_characters=5, min_fraction_letters=.5):
    if max_sentences:
        if isinstance(max_sentences, int):
            max_sentences = min(max_sentences / len(sentences), 1.0)
        min_length = np.quantile([len(sentence) for sentence in sentences], 1-max_sentences)
        min_characters = max(min_characters, min_length)
        
    sentences = filter(lambda sentence: _is_meaningfull_sentence(sentence, min_characters, min_fraction_letters), sentences)
    return list(sentences)
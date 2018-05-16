import string

from nltk import pos_tag
from nltk.corpus import stopwords


def text_process(text):
    no_punc = [char for char in text if char not in string.punctuation]
    no_punc = ''.join(no_punc)

    words = []
    for word in no_punc.split():
        if word.lower() not in stopwords.words('english'):
            words.append(word)
            words.append(pos_tag([word], tagset="universal")[0][1])

    return words

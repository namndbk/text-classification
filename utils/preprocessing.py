import string

from ftfy import fix_text


def normalize_text(text):
    return text


def normalize(text):
    text = fix_text(text)
    text = " ".join(i for i in text.split())
    table = str.maketrans({key: None for key in string.punctuation})
    text = text.translate(table)
    return text.lower()


def load_dataset(path):
    pass
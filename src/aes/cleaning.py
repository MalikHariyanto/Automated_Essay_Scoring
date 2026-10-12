import unicodedata

POLICY = 'NFC-newline-control-v1'

def clean(text, normalize_punctuation=False):
    text = unicodedata.normalize('NFC', text).replace('\r\n','\n').replace('\r','\n')
    text = ''.join(c for c in text if c in '\n\t' or unicodedata.category(c) != 'Cc')
    if normalize_punctuation:
        text = text.translate(str.maketrans({'“':'"','”':'"','‘':"'",'’':"'",'–':'-','—':'-'})).replace('…','...')
    return text

import re

# Code for the utils.py file is currently from 
# https://github.com/Montinger/Transformer-Workbench/blob/main/transformer-from-scratch/common.py

# python mapping dicts
# The following mapping dicts are used to cleanse texts later
# They are used by helper functions further below
remap_dict_1 = {
    '„ ' : '"', # fix non-aligned beginnings
    ' “' : '"', # fix non-aligned beginnings
    '\u0093' : '"',
    '\u0094' : '"',
    '\u0097' : ' ',
    ' “' : '"', # fix non-aligned beginnings
    '\u00a0' : ' ', # non-breaking white space
    '\u202f' : ' ', # narrow non-breaking white space
    'Ã¶' : 'ö', # german oe
    'Ã¼' : 'ü', # german ue
    'Ã¤' : 'ä', # german ae
}

remap_dict_2 = {
    '„'  : '"',
    '“'  : '"',
    '‟'  : '"',
    '”'  : '"',
    '″'  : '"',
    '‶'  : '"',
    '”'  : '"',
    '‹'  : '"',
    '›'  : '"',
    '’'  : "'",
    '′'  : "'",
    '′'  : "'",
    '‛'  : "'",
    '‘'  : "'",
    '`'  : "'",
    '–'  : '--',
    '‐'  : '-',
    '»'  : '"',
    '«'  : '"',
    '≪'  : '"',
    '≫'  : '"',
    '》' : '"',
    '《' : '"',
    '？' : '?',
    '！' : '!',
    '…'  : ' ... ',
    '\t' : ' ',
    '。' : '.', # chinese period
    '︰' : ':',
    '〜' : '~',
    '；' : ';',
    '）' : ')',
    '（' : '(',
    'ﬂ'  : 'fl', # small ligature characters
    'ﬁ'  : 'fi',
    '¶'  : ' ',
}

filter_unicode_ranges_dict = {
    'chinese'    : "\u4e00-\u9fff",
    'japanese'   : "\u3040-\u309f", # Hiragana
    'japanese2'  : "\u30a0-\u30ff", # Hiragana
    'cyrillic'   : "\u0400-\u04ff",
    'devanagari' : "\u0900-\u0954", # hindi is here
    'korean1'    : "\uac00-\ud7a3",
    'korean2'    : "\u1100-\u11ff",
    'korean3'    : "\u3130-\u318f",
    'korean4'    : "\ua960-\ua97f",
    'korean5'    : "\ud7b0-\ud7ff",
    'malayalam'  : "\u0d00-\u0d7f",
    'arabic1'    : "\u0600-\u06ff",
    'arabic2'    : "\u0750-\u077f",
    'arabic3'    : "\u0870-\u089f",
    'arabic4'    : "\u08a0-\u08ff",
    'arabic5'    : "\ufb50-\ufdff",
    'arabic6'    : "\ufe70-\ufeff",
    'hebrew'     : "\u0590-\u05ff",
    'ethiopic'   : "\u1200-\u137f",
    'chinese1'   : "\u4e00-\u4fff",
    'chinese2'   : "\u5000-\u57ff",
    'chinese3'   : "\u5800-\u5fff",
    'chinese4'   : "\u6000-\u67ff",
    'chinese5'   : "\u6800-\u6fff",
    'chinese6'   : "\u7000-\u77ff",
    'chinese7'   : "\u7800-\u7fff",
    'chinese8'   : "\u8000-\u87ff",
    'chinese9'   : "\u8800-\u8fff",
    'chinese10'  : "\u9000-\u97ff",
    'chinese11'  : "\u9800-\u9fff",
    'chinese12'  : "\u3100-\u312f",
    'chinese13'  : "\u31a0-\u31bf",
    'glagolitic1': "\u2c00-\u2c5f",
    'bengali'    : "\u0980-\u09ff",
    'telugu'     : "\u0c00-\u0c7f",
    'rumi'       : "\U00010e60-\U00010e7e",
    'arabic7'    : "\U00010ec0-\U00010eff",
    'indic'      : "\U0001ec70-\U0001ecbf",
    'ottoman'    : "\U0001ed00-\U0001ed4f",
    'arabic-m'   : "\U0001ee00-\U0001eeff",
    'glagolitic2': "\U0001e000-\U0001e02f",

}



def remove_unicode_range(text, unicode_range):
    """
    Remove a specified Unicode range from a given text and replace it with a placeholder.

    Parameters
    ----------
    text : str
        The input text from which the specified Unicode range needs to be removed.
    unicode_range : str
        The range of Unicode characters to remove from the input text.

    Returns
    -------
    str
        The text with the specified Unicode range removed and replaced with the placeholder '\[UNK\]'.

    Example
    -------
    >>> sample = 'I am from 美国。We should be friends. 朋友。'
    >>> remove_unicode_range(sample, '\u4e00-\u9fff')
    'I am from \[UNK\]We should be friends. \[UNK\]'
    """
    # print(unicode_range.encode('unicode-escape'), str) # for debugging
    return re.sub(rf'[{unicode_range}]+', '\[UNK\]', text, flags=re.U)



def cleanse_text(text, remove_unicode=True):
    """
    Cleans a text, removing special characters and optionally Unicode characters.

    Parameters
    ----------
    text : str
        The input text to be cleaned.
    remove_unicode : bool, optional
        Whether to remove Unicode characters from the text (default is True).

    Returns
    -------
    str
        The cleaned text.
    """
    for old, new in remap_dict_1.items():
        text = text.replace(old, new)
    for old, new in remap_dict_2.items():
        text = text.replace(old, new)
    if remove_unicode:
        for key, unicode_range in filter_unicode_ranges_dict.items():
            text = remove_unicode_range(text, unicode_range)

    # remove double spaces
    while text.find('  ') >= 0:
        text = text.replace('  ', ' ').replace('  ', ' ')

    return text


def cleans_test_text(text):
    """
    Cleans the weird format for the test set to be like the other datasets by removing
    additional HTML-like tags.

    Parameters
    ----------
    text : str
        The input text to be cleaned.

    Returns
    -------
    str
        The cleaned text with HTML-like tags removed.
    """
    text = text.replace('</seg>', '')
    text = text.replace('</doc>', '')
    text = text.replace('</refset>', '')
    text = re.sub(rf'<seg id="[0-9]+">', '', text, flags=re.U)
    text = re.sub(rf'<doc .+?>', '', text, flags=re.U)
    text = re.sub(rf'<refset .+?>', '', text, flags=re.U)
    while text.find('\n\n') >= 0:
        text = text.replace('\n\n', '\n')
    return text

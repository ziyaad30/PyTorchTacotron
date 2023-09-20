import re
import os
from text import cleaners
from text.symbols import symbols


# load phonemizer
from phonemizer.backend import EspeakBackend
if os.name == 'nt':
    from phonemizer.backend.espeak.wrapper import EspeakWrapper
    _ESPEAK_LIBRARY = 'C:\Program Files\eSpeak NG\libespeak-ng.dll'    # For Windows
    EspeakWrapper.set_library(_ESPEAK_LIBRARY)

# Mappings from symbol to numeric ID and vice versa:
symbol_to_id = {s: i for i, s in enumerate(symbols)}
id_to_symbol = {i: s for i, s in enumerate(symbols)}

# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r'(.*?)\{(.+?)\}(.*)')


def phoneme_text(text):
    backend = EspeakBackend(language='en-us', preserve_punctuation=True, with_stress=False, punctuation_marks=';:,.!?¡¿—…"«»“”()', language_switch='remove-flags')
    text = backend.phonemize([text], strip=True)[0]
    return text.strip()

def phoneme_to_sequence(text):
    sequence = []

    for symbol in text:
        if symbol in symbol_to_id.keys():
            symbol_id = symbol_to_id[symbol]
            sequence += [symbol_id]
        else:
            raise Exception(f"Sorry, symbol {symbol} not found!")
    
    # Append EOS token
    # sequence.append(symbol_to_id['~'])

    return sequence

def text_to_sequence(text, cleaner_names):
    sequence = []
    
    # Clean text
    clean_text = _clean_text(text, cleaner_names)

    # Phonemize text
    text = phoneme_text(text)
    
    # Convert text to symbols
    for symbol in text:
        if symbol in symbol_to_id.keys():
            symbol_id = symbol_to_id[symbol]
            sequence += [symbol_id]
        else:
            raise Exception(f"Sorry, symbol {symbol} not found!")

    # Append EOS token
    # sequence.append(symbol_to_id['~'])
    return sequence


def sequence_to_text(sequence):
    '''Converts a sequence of IDs back to a string'''
    result = ''
    for symbol_id in sequence:
        if symbol_id in id_to_symbol:
            s = id_to_symbol[symbol_id]
            result += s
    return result.replace('}{', ' ')


def _clean_text(text, cleaner_names):
  for name in cleaner_names:
    cleaner = getattr(cleaners, name)
    if not cleaner:
      raise Exception('Unknown cleaner: %s' % name)
    text = cleaner(text)
  return text

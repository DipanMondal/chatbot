import json
import numpy as np
import re

# upto v-2
qs_max = 130
ans_max = 377

with open("vocab1.json","r") as file:
    vocab = json.load(file)
    
with open("contractions.json","r") as file:
    contractions_dict = json.load(file)
    
vocabulary = {i:w for w,i in zip(vocab.keys(),vocab.values())}

contractions_re = re.compile('(%s)' % '|'.join(re.escape(key) for key in contractions_dict.keys()), re.IGNORECASE)

def expand_contractions(sentence, contractions_dict=contractions_dict):
    def replace(match):
        # Match is case-insensitive, use the original case in replacement
        contraction = match.group(0)
        expanded = contractions_dict.get(contraction.lower())
        if contraction[0].isupper():
            expanded = expanded.capitalize()
        return expanded
    return contractions_re.sub(replace, sentence)

def Word2Num(word):
    try:
        return vocab[word]
    except:
        return -1

def Sent2Seq(sentence):
    sentence = expand_contractions(sentence.lower())
    sentence = re.sub(r"""([+$@#%^&.?!*"\\',:;-])""", r' \1 ', sentence)
    tokens = sentence.strip().split()
    return list(map(Word2Num,tokens))

def padding(sequence:list,max_pad:int):
    l = max_pad-len(sequence)
    for i in range(l):
        sequence.append(0)
        
def preprocess_input(input_sentence):
    seq = Sent2Seq(input_sentence)
    padding(seq,qs_max)
    return np.array(seq)
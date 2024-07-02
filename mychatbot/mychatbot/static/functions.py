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
    
def QandA(enc_model,dec_model,vocabulary,preprocess_input,sentence):
    states_values = enc_model.predict(np.array([preprocess_input(sentence)]))
    empty_target_seq = np.zeros((1 , 1))
    empty_target_seq[0, 0] = vocab['<start>']
    stop_condition = False
    decoded_translation = ''
    
    while not stop_condition :
        dec_outputs , h , c = dec_model.predict([empty_target_seq] + states_values)
        sampled_word_index = np.argmax(dec_outputs[0, -1, :])
        sampled_word = None
        
        word = vocabulary[sampled_word_index]
        decoded_translation += f' {word}'
        sampled_word = word
        
        if sampled_word == '<end>' or len(decoded_translation.split()) > ans_max:
            stop_condition = True
            
        empty_target_seq = np.zeros((1 , 1))  
        empty_target_seq[0 , 0] = sampled_word_index
        states_values = [h , c] 
    ans = decoded_translation.replace("<end>","")
    return ans
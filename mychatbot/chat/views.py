from django.shortcuts import render

# Create your views here.

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from django.shortcuts import render, redirect
from tensorflow.keras.models import load_model
from django.conf import settings
from django.contrib.staticfiles import finders

# imp functions
import json
import numpy as np
import re


class ChatBot:
    def __init__(self):
        # upto v-2
        self.qs_max = 130
        self.ans_max = 377

        vocab_path = finders.find('vocab1.json')
        with open(vocab_path, "r") as file:
            self.vocab = json.load(file)

        contractions_path = finders.find('contractions.json')
        with open(contractions_path, "r") as file:
            self.contractions_dict = json.load(file)

        self.vocabulary = {i: w for w, i in zip(self.vocab.keys(), self.vocab.values())}

        self.contractions_re = re.compile('(%s)' % '|'.join(re.escape(key) for key in self.contractions_dict.keys()), re.IGNORECASE)

    def expand_contractions(self,sentence):
        def replace(match):
            # Match is case-insensitive, use the original case in replacement
            contraction = match.group(0)
            expanded = self.contractions_dict.get(contraction.lower())
            if contraction[0].isupper():
                expanded = expanded.capitalize()
            return expanded

        return self.contractions_re.sub(replace, sentence)

    def Word2Num(self,word):
        try:
            return self.vocab[word]
        except:
            return -1

    def Sent2Seq(self,sentence):
        sentence = self.expand_contractions(sentence.lower())
        sentence = re.sub(r"""([+$@#%^&.?!*"\\',:;-])""", r' \1 ', sentence)
        tokens = sentence.strip().split()
        return list(map(self.Word2Num, tokens))

    def padding(self,sequence: list, max_pad: int):
        l = max_pad - len(sequence)
        for i in range(l):
            sequence.append(0)

    def preprocess_input(self,input_sentence):
        seq = self.Sent2Seq(input_sentence)
        self.padding(seq, self.qs_max)
        return np.array(seq)

    def QandA(self,enc_model, dec_model, sentence):
        states_values = enc_model.predict(np.array([self.preprocess_input(sentence)]), verbose=0)
        empty_target_seq = np.zeros((1, 1))
        empty_target_seq[0, 0] = self.vocab['<start>']
        stop_condition = False
        decoded_translation = ''

        while not stop_condition:
            dec_outputs, h, c = dec_model.predict([empty_target_seq] + states_values, verbose=0)
            sampled_word_index = np.argmax(dec_outputs[0, -1, :])
            sampled_word = None

            word = self.vocabulary[sampled_word_index]
            decoded_translation += f' {word}'
            sampled_word = word

            if sampled_word == '<end>' or len(decoded_translation.split()) > self.ans_max:
                stop_condition = True

            empty_target_seq = np.zeros((1, 1))
            empty_target_seq[0, 0] = sampled_word_index
            states_values = [h, c]
        ans = decoded_translation.replace("<end>", "")
        return ans


# models
p = finders.find('Encoder2.h5')
enc_model = load_model(p)
p = finders.find('Decoder2.h5')
dec_model = load_model(p)

bot = ChatBot()

def home(request):
    return render(request,'index.html')

@csrf_exempt
def chat_message(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        message = data.get('message')

        # Here, you can implement your chatbot logic to generate a response
        try:
            ans = bot.QandA(enc_model,dec_model,message)
        except:
            ans="sorry ! i don't have the answer ." 
        response = f"Bot: {ans}"

        return JsonResponse({'response': response})
    return JsonResponse({'error': 'Invalid request'}, status=400)

from keras.models import model_from_json
from keras.callbacks import LambdaCallback, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import io
import time

import serial

# 生成元の文章を読み込む
path = './txt/alice.txt'
with io.open(path, encoding='utf-8') as f:
    text = f.read().lower()
print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

# モデルを読み込む
model = model_from_json(open('./complete_model_epoch_30.json').read())
# 学習結果を読み込む
model.load_weights('./complete_model_epoch_30.h5')

model.summary()

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

start_index = random.randint(0, len(text) - maxlen - 1)


def generate(sentence, length, _diversity):
    for diversity in [_diversity]:
        print('----- diversity:', diversity)

        generated = ''
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        print()
        # sys.stdout.write(generated)

        for i in range(length):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

    return generated


def sample(preds, temperature=0.9):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


if __name__ == '__main__':
    seed = 'get very tired of the door bein the shap'

    ser = serial.Serial("/dev/cu.usbserial-A105A98D", 9600)
    time.sleep(1)

    while True:
        generated = generate(seed, 500, 0.55)
        # generated = generated.replace('\n', '').replace('\t', '')
        # 空白の連続を削除
        generated = ' '.join(generated.split())
        generated = generated[40:]
        print(generated)
        ser.write(generated.encode('utf-8'))      # シリアルポートに出力

        # seedを更新
        seed = generated[-40:]

        time.sleep(60)
        print()

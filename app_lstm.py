from keras.layers import Dense, Activation, LSTM, Dropout
from keras.models import Sequential
import numpy as np

FILE_PATH = "./alice.txt"
text = ""
with open(FILE_PATH, 'r', encoding="utf-8") as f:
    for line in f:
        lines = line.split()
        text += " ".join(lines)
text = text.lower()
print(len(text))

chars = set(text)
nb_chars = len(chars)
char2index = dict((c, i) for i, c in enumerate(chars))
index2char = dict((i, c) for i, c in enumerate(chars))

SEQLEN = 10
STEP = 1

input_chars = []
label_chars = []

for i in range(0, len(text) - SEQLEN, STEP):
    input_chars.append(text[i:i+SEQLEN])
    label_chars.append(text[i+SEQLEN])

X = np.zeros((len(input_chars), SEQLEN, nb_chars), dtype=np.bool)
Y = np.zeros((len(input_chars), nb_chars), dtype=np.bool)
for i, input_char in enumerate(input_chars):
    for j, ch in enumerate(input_char):
        X[i, j, char2index[ch]] = 1
    Y[i, char2index[label_chars[i]]] = 1

HIDDEN_SIZE = 128
BATCH_SIZE = 128
NUM_ITERATIONS = 50
NUM_EPOCHS_PER_ITERATION = 1
NUM_PREDS_PER_EPOCH = 100

model = Sequential()
model.add(LSTM(HIDDEN_SIZE, return_sequences=False,
               input_shape=(SEQLEN, nb_chars), unroll=True))
model.add(Dropout(0.2))
model.add(Dense(nb_chars))
model.add(Activation("softmax"))
model.compile(loss="categorical_crossentropy", optimizer="rmsprop")

for iteration in range(NUM_ITERATIONS):
    print("=" * 50)
    print("Iteration #: %d" % (iteration))
    history = model.fit(X, Y, batch_size=BATCH_SIZE,
                        epochs=NUM_EPOCHS_PER_ITERATION)
    test_index = np.random.randint(len(input_chars))
    test_chars = input_chars[test_index]
    print("Generating form sead: {}".format(test_chars))
    print(test_chars, end="")
    for i in range(NUM_PREDS_PER_EPOCH):
        Xtest = np.zeros((1, SEQLEN, nb_chars))
        for i, ch in enumerate(test_chars):
            Xtest[0, i, char2index[ch]] = 1
        pred = model.predict(Xtest, verbose=0)[0]
        ypred = index2char[np.argmax(pred)]
        print(ypred, end="")
        test_chars = test_chars[1:] + ypred
    print()

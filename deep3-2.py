from keras.models import Sequential
from keras import layers
from keras.preprocessing.text import Tokenizer
import pandas as pd
from keras.layers.core import Flatten
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.layers import Embedding
from keras.preprocessing.sequence import pad_sequences


df = pd.read_csv('imdb_masters.csv',encoding='latin-1')
print(df.head())
sentences = df['review'].values
y = df['label'].values
max_review_len= max([len(s.split()) for s in sentences])


#tokenizing data
tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(sentences)
#getting the vocabulary of data
sentences = tokenizer.texts_to_sequences(sentences)

padded_docs= pad_sequences(sentences,maxlen=max_review_len)
vocab_size= len(tokenizer.word_index)+1

le = preprocessing.LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(padded_docs, y, test_size=0.25, random_state=1000)

# Number of features
# print(input_dim)
model = Sequential()
model.add(Embedding(vocab_size, 50, input_length=max_review_len))
model.add(Flatten())
model.add(layers.Dense(300, activation='relu'))

model.add(layers.Dense(3, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['acc'])
history=model.fit(X_train,y_train, epochs=2, verbose=True, validation_data=(X_test,y_test), batch_size=256)
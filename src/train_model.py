#a model trained on a subset of the comments section of the data

# data processing tools
import string, os 
import pandas as pd
import numpy as np
np.random.seed(42)

# keras module for building LSTM 
import tensorflow as tf
tf.random.set_seed(42)
import tensorflow.keras.utils as ku 
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# surpress warnings
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

#i am using some of the helper functions from session 8 to clean and prep the data
def clean_text(txt):
    if isinstance(txt, str):
        txt = "".join(v for v in txt if v not in string.punctuation).lower()
        txt = txt.encode("utf8").decode("ascii",'ignore')
        return txt 
    else:
        return ""

def get_sequence_of_tokens(tokenizer, corpus):
    ## convert data to sequence of tokens 
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)
    return input_sequences

def generate_padded_sequences(input_sequences, total_words):
    # get the length of the longest sequence
    max_sequence_len = max([len(x) for x in input_sequences])
    # make every sequence the length of the longest on
    input_sequences = np.array(pad_sequences(input_sequences, 
                                            maxlen=max_sequence_len, 
                                            padding='pre'))

    predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
    label = ku.to_categorical(label, 
                            num_classes=total_words)
    return predictors, label, max_sequence_len

def create_model(max_sequence_len, total_words):
    input_len = max_sequence_len - 1
    model = Sequential()
    
    # Add Input Embedding Layer
    model.add(Embedding(total_words, 
                        10, 
                        input_length=input_len))
    
    # Add Hidden Layer 1 - LSTM Layer
    model.add(LSTM(100))
    model.add(Dropout(0.1))
    
    # Add Output Layer
    model.add(Dense(total_words, 
                    activation='softmax'))

    model.compile(loss='categorical_crossentropy', 
                    optimizer='adam')
    
    return model

# defining the path and loading a subset of the data
data_dir = os.path.join("431868/news_data")
filename = "CommentsJan2018.csv"

# Loading 1000 comments from one of the files in the dataset
comments_df = pd.read_csv(data_dir + "/" + filename)
comments_subset = comments_df["commentBody"].values[:1000]  # Extracting the first 1000 comments

#cleaning up the data
comments_subset = [comment for comment in comments_subset if comment != "Unknown"]
# creating corpus
corpus = [clean_text(comment) for comment in comments_subset]

#tokenizing the data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

# turning comment input into numericals
inp_sequences = get_sequence_of_tokens(tokenizer, corpus)

#padding input sequences
predictors, label, max_sequence_len = generate_padded_sequences(inp_sequences, total_words)

#creating model
model = create_model(max_sequence_len, total_words)
history = model.fit(predictors, 
                    label, 
                    epochs=50,
                    batch_size=128, 
                    verbose=1)

# saving the model in "out" folder
out_dir = "assignment-3---rnns-for-text-generation-VictoriaTrabW/out"

save_path = os.path.join(out_dir, "trained_model")
tf.keras.saving.save_model(model, save_path)

#saving the tokenizer as json for later use in the generate_text function in the text_gen.py script
tokenizer_path = os.path.join(out_dir, "tokenizer.json")
tokenizer_json = tokenizer.to_json()
with open(tokenizer_path, "w") as json_file:
    json_file.write(tokenizer_json)
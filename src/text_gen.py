#the trained model is loaded and then text can be generated from a user-suggested prompt

import os
import tensorflow as tf

# Loading the model which was trained in the train_model.py script
model_path = "assignment-3---rnns-for-text-generation-VictoriaTrabW/out/trained_model"
loaded_model = tf.keras.models.load_model(model_path)

# Loading the tokenizer from JSON
tokenizer_path = "assignment-3---rnns-for-text-generation-VictoriaTrabW/out/tokenizer.json"
with open(tokenizer_path, "r") as json_file:
    tokenizer_json = json_file.read()
tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json)


def generate_text(seed_text, next_words, model, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], 
                                    maxlen=max_sequence_len-1, 
                                    padding='pre')
        predicted = np.argmax(model.predict(token_list),
                                            axis=1)
        
        output_word = ""
        for word,index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " "+output_word
    return seed_text.title()
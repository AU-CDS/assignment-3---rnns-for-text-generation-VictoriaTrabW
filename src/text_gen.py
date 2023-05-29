# in this script, text can be generated from a user-suggested prompt
import os
import tensorflow as tf

# Loading the model which was trained in the train_model.py script
model_path = "assignment-3---rnns-for-text-generation-VictoriaTrabW/out/trained_model"
loaded_model = tf.keras.models.load_model(model_path)

# Loading the tokenizer from the out folder 
tokenizer_path = "assignment-3---rnns-for-text-generation-VictoriaTrabW/out/tokenizer.json"
with open(tokenizer_path, "r") as json_file:
    tokenizer_json = json_file.read()
tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json)

# Calculating max sequence length
max_sequence_len = tokenizer.get_config()['max_sequence_length']

#this is one of the helper functions from session 8, which can generate text
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

# the text generator in use
seed_prompt = "stop"
num_words = 10

generated_text = generate_text(seed_prompt, num_words, loaded_model, max_sequence_len)
print(generated_text)
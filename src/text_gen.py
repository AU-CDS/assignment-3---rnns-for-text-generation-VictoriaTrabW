#the trained model is loaded and then text can be generated from a user-suggested prompt

import os
import tensorflow as tf

#loading the model which was trained in the train_model.py script
model_path = "assignment3-rnns-for-text-generation-VictoriaTrabW/out/trained_model"
loaded_model = tf.keras.models.load_model(model_path)

# Function to generate text from user input
def generate_text_from_input(loaded_model, max_sequence_len, tokenizer):
    while True:
        # Prompt user for input
        prompt = input("Enter a prompt (or 'q' to quit): ")
        if prompt.lower() == 'q':
            break
        
        # Generate text based on the user input
        generated_text = generate_text(prompt, 7, model, max_sequence_len, tokenizer)
        print("Generated text:", generated_text)

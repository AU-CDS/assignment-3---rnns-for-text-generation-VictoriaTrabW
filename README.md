[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-8d59dc4de5201274e310e4c54b9627a8934c3b88527886e3b421487c677d23eb.svg)](https://classroom.github.com/a/5f7lMH9Y)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-c66648af7eb3fe8bc4f294546bfd86ef473780cde1dea487d3c4ff354943c9ae.svg)](https://classroom.github.com/online_ide?assignment_repo_id=10587060&assignment_repo_type=AssignmentRepo)
# Assignment 3 - Language modelling and text generation using RNNs

Text generation is hot news right now!

For this assignemnt, you're going to create some scripts which will allow you to train a text generation model on some culturally significant data - comments on articles for *The New York Times*. You can find a link to the data [here](https://www.kaggle.com/datasets/aashita/nyt-comments).

You should create a collection of scripts which do the following:

- Train a model on the Comments section of the data
  - [Save the trained model](https://www.tensorflow.org/api_docs/python/tf/keras/models/save_model)
- Load a saved model
  - Generate text from a user-suggested prompt

## Objectives

Language modelling is hard and training text generation models is doubly hard. For this course, we lack somewhat the computationl resources, time, and data to train top-quality models for this task. So, if your RNNs don't perform overwhelmingly, that's fine (and expected). Think of it more as a proof of concept.

- Using TensorFlow to build complex deep learning models for NLP
- Illustrating that you can structure repositories appropriately
- Providing clear, easy-to-use documentation for your work.

## Some tips

One big thing to be aware of - unlike the classroom notebook, this assignment is working on the *Comments*, not the articles. So two things to consider:

1) The Comments data might be structured differently to the Articles data. You'll need to investigate that;
2) There are considerably more Comments than articles - plan ahead for model training!

## Additional pointers

- Make sure not to try to push the data to Github!
- *Do* include the saved models that you output
- Make sure to structure your repository appropriately
  - Include a readme explaining relevant info
    - E.g where does the data come from?
    - How do I run the code?
- Make sure to include a requirements file, etc...

## Repository structure
- This repository does not contain an "in" folder, as the data comes from our shared resources, and the data should not be pushed back into Github due to its size.

- **out** folder: Contains the outputs generated during the execution of the scripts. This includes a trained model and a tokenizer.

- **src** folder: Contains the main scripts to solve the assignment. This includes scripts for training the model, loading a saved model, and generating text.

- **requirements.txt** file: Lists the required programs and packages to run the code. You can use it to install the necessary dependencies. Can be installed with: pip install -r requirements.txt

- **README.md** file: Contains the assignment details, dependencies, additional notes, and reflections on the output. It provides clear and easy-to-understand documentation for the project.

## Dependencies and data
The project has been run through UCloud in the Coder Python app (1.78.2), and the neccesary programs are listed in requirements.txt.

The data comes from our shared data folder in language analytics (called 431868 in my assignment). The files are in a folder called news_data and consist of .csv data about article names and comments from several months in 2017-2018. This is a link for the data https://www.kaggle.com/datasets/aashita/nyt-comments

## Reflections and methods
Due to limited computational resources (script being killed) and time constraints, I worked with a smaller subset of the data to demonstrate the proof of concept. However, the provided code can be scaled up to handle the entire dataset if sufficient computational capacity is available.

To build the text generation model, I used a recurrent neural network (RNN) architecture, specifically the Long Short-Term Memory (LSTM) model, implemented using TensorFlow's Keras API. The steps to train the model involved data cleaning, tokenization, input sequence generation, model construction, training, and model saving.

It is important to note that the model's performance in generating text depends on various factors, such as the size and complexity of the dataset, which in my case has been limited as stated above.


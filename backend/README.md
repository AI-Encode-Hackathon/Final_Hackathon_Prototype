# Machine Learning Model Trainer

---

The goal of this project was to create a low code solution for non-programmers or people with little knowledge of machine learning. Non-programmers will be able to train basic models and receive the parameters used to train the model as well as statistics on its performance. This gives the user a starting point for development.

This project was undertake at the AI Encode Club Hackathon 2024.

## Technologies Used
Frontend:
- The frontend User Interface has been programmed using Internet Computer and React.

Backend:
- The backend has been programmed using python and fastAPI.

## How to run
Run the backend by first installing the requirements:
```commandline
pip install -r requirements.txt
```

Then run fastAPI inside the backend directory:
```commandline
uvicorn main:app --host 0.0.0.0 --port 8001
```

Run the frontend in the frontend directory:
```commandline
dfx deploy
```

## Input / Output
The user uploads a csv file with the columns "label" and "path". The label is the expected output and the path is the path to either the .jpg or .mp4 file associated with it.

They then have the choice of entering advanced options:
 - training_method : specific learning method to classify the input e.g k-nearest neighbours
 - training_type : type of learning method i.e. supervised/unsupervised
 - learning_rate : default value set to 0.001. In the case of k-means or k-nearest neighbours, the default value is 5
 - threshold : minimum accuracy to return the class

The output will be:
 - Accuracy of the model (does not apply to unsupervised learning)
 - Learning method
 - Training time
 - Threshold
 - Learning rate / k-value

## Advantages of this solution
Through using Inernet Computer, the solution is decentralised and scaleable.

It allows users with little knowledge of machine learning to easily create a starting solution and adjust basic parameters.

## Future Developments
Future developments for this project include:
 - Adding additional training algorithms
 - Adding additional parameters
 - Allowing additional data formats

from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from model_trainers import k_means, k_nearest_neighbours, linear_regression, mlp, decision_tree
from pre_process_audio import create_audio_embedding
from pre_process_images import create_image_embedding
from tqdm import tqdm
import torch

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])


class Response(BaseModel):
    accuracy: str
    training_method: str
    training_time: str
    threshold: float
    hyperparameter: int

@app.post("/train_model")
def train_model(labels: UploadFile, learning_class: str="Supervised",training_method: str="Regression", k: int=7) -> Response:
    start = datetime.now()

    print("loading csv")
    labels_and_paths = pd.read_csv(labels.file)

    # Create embeddings for training data
    print("creating embeddings")
    embeddings = []
    classes = []

    pbar = tqdm(desc="create embeddings", total=14493)
    for label, path in zip(labels_and_paths["label"], labels_and_paths["path"]):
        extension = "".join(path.split(".")[1:]).lower()
        try:
            if extension == "wav":
                wav2mel = torch.jit.load("wav2mel.pt")
                dvector = torch.jit.load("dvector.pt").eval()
                emb = create_audio_embedding(path, wav2mel, dvector)
            elif extension in ["jpg", "jpeg"]:
                emb = create_image_embedding(path)
            else:
                return {
                    "accuracy": "Invalid data file type. Must be jpg or wav",
                    "training_method": training_method,
                    "training_time": "n/a",
                    "threshold": 0.7,
                    "hyperparameter": int(k)
                }

            embeddings.append(emb)
            classes.append(label)
            pbar.update(1)

        except FileNotFoundError:
            print("path does not exist")
            continue


    print("selecting training method")
    if learning_class == "Unsupervised":
        model = k_means.train_model(k, embeddings)
    else:
        # Split dataset into test and train
        print("splitting dataset")
        X_train, X_test, y_train, y_test = train_test_split(embeddings, classes, test_size=0.2, random_state=42)

        match training_method:
            case "K Nearest Neighbours":
                model = k_nearest_neighbours.train_model(k, X_train, y_train)
            case "MLP":
                k=0
                model = mlp.train_model(X_train, y_train)
            case "regression":
                k=0
                model = linear_regression.train_model(X_train, y_train)
            case "Decision Tree":
                k = 0
                model = decision_tree.train_model(X_train, y_train)

    end = datetime.now()
    time_taken = f"{end - start}"
    print("evaluating model")

    if learning_class == "Supervised":
        try:
            accuracy = model.score(X_test, y_test)
            accuracy = format(accuracy, ".0%")
        except UnboundLocalError:
            accuracy = "Invalid training method"
    else:
        accuracy = "n/a"
    print(accuracy)


    return {
        "accuracy": str(accuracy),
        "training_method": training_method,
        "training_time": time_taken,
        "threshold": 0.7,
        "hyperparameter": int(k)
    }


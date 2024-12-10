from io import StringIO
import pickle

import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile
from pydantic import BaseModel

with open("data/ridge_serialized.pickle", "rb") as f:
    model = pickle.load(f)

with open("data/onehot_serialized.pickle", "rb") as f:
    encoder = pickle.load(f)


app = FastAPI()

class Item(BaseModel):
    name: str
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: int
    engine: int
    max_power: int
    seats: float


def preprocessing(data, encoder):
    cat_cols = ['name', 'fuel', 'seller_type', 'transmission', 'owner', 'seats']

    X_train_cat = data[cat_cols]
    X_train_num = data.drop(cat_cols, axis = 1)

    X_train_cat['seats'] = X_train_cat['seats'].astype(int).astype(str)
    X_encoded = encoder.transform(X_train_cat)

    X = np.concatenate((X_encoded, X_train_num.values), axis = 1)
    return X

@app.post("/predict_item")
async def predict_item(
    item: Item, 
) -> float:
    data = pd.DataFrame.from_dict([dict(item)])

    X = preprocessing(data, encoder)

    prediction = model.predict(X)[0]
    return prediction


@app.post("/predict_items")
async def predict_items(
    file: UploadFile, 
):
    contents = await file.read()
    data = pd.read_csv(StringIO(contents.decode("utf-8")))
    
    X = preprocessing(data, encoder)

    predictions = model.predict(X)
    data["predicted_price"] = predictions
    data.to_csv('data/output.csv', index=False)

    output = StringIO()
    data.to_csv(output, index=False)
    output.seek(0)
    return {"message": "Предсказания выполнены", "csv": output.getvalue()}

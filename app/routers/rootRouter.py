from fastapi import APIRouter , File , UploadFile , Form
import os
import io
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import pandas as pd
from app.helpers.preprocess import preprocessEncoding , one_hot_encoding
from app.models.train import   train
from app.models.test import test

router = APIRouter()


# @router.post("/preprocess")
# def question(data):    
#     df = preprocess
# @router.post("/grant")
# async def grant_matching(data: dict):

@router.get("/preprocess")
async def preprocess_data():
    # Add your data preprocessing logic here
    preprocessEncoding()
    return {"message": "Data processed"}

@router.post("/train")
async def train_model1(lr:str = Form(...) , epochs:str = Form(...) , variable:str = Form(...) , split:str = Form(...) , data: UploadFile = File(...) ):
    # Add your model training logic here
    lr = float(lr)
    epochs = int(epochs)
    split = int(split)
    print(lr , epochs , variable ,split)
    contents = await data.read()
    contents = io.StringIO(contents.decode('utf-8'))
    try:
        df = pd.read_csv(contents)
        # return JSONResponse(content={"data": df.to_dict(orient="records")}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    print(df)
    preprocessed_df = preprocessEncoding(df)
    print(preprocessed_df)
    mse = train(preprocessed_df , lr , epochs ,variable  , split)
    print(mse)
    return "model training completed and model saved as model.pth"

@router.post("/test")
async def test_model(data: UploadFile = File(...)):
    # Add your model training logic here
    contents = await data.read()
    contents = io.StringIO(contents.decode('utf-8'))
    try:
        df = pd.read_csv(contents)
        # return JSONResponse(content={"data": df.to_dict(orient="records")}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    preprocessed_df = preprocessEncoding(df)
    predictions = test(preprocessed_df)
    list1 = predictions.tolist()
    print(type(list1))
    return jsonable_encoder(list1)

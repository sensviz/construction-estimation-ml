from fastapi import APIRouter , File , UploadFile
import os
import io
from fastapi.responses import JSONResponse
import pandas as pd
from app.helpers.preprocess import preprocessEncoding
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
async def train_model1(data: UploadFile = File(...)):
    # Add your model training logic here
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
    train(preprocessed_df)
    return {"message": "Model training endpoint"}

@router.get("/test")
async def test_model():
    # Add your model training logic here
    test()
    return {"message": "Model testing endpoint"}

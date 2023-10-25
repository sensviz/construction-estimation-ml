from fastapi import APIRouter , File , UploadFile
import os
import io
from fastapi.responses import JSONResponse
import pandas as pd
from app.helpers.preprocess import preprocessEncoding
from app.models.train import train_model

router = APIRouter()


# @router.post("/preprocess")
# def question(data):    
#     df = preprocess
# @router.post("/grant")
# async def grant_matching(data: dict):

@router.get("/preprocess")
async def preprocess_data():
    # Add your data preprocessing logic here
    preprocess()
    return {"message": "Data processed"}

@router.post("/train")
async def train_model1(data: UploadFile = File(...)):
    # Add your model training logic here
    print(data)
    name, extension = os.path.splitext(data.filename)
    print(extension)
    print(data)
    contents = await data.read()
    contents = io.StringIO(contents.decode('utf-8'))
    try:
        df = pd.read_csv(contents)
        # return JSONResponse(content={"data": df.to_dict(orient="records")}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    print(df)
    print(extension)
    preprocessed_df = preprocessEncoding(df)
    print(preprocessed_df)
    train_model(preprocessed_df)
    return {"message": "Model training endpoint"}

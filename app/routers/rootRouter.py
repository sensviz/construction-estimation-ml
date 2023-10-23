from fastapi import APIRouter
import os
from helpers.preprocess import preprocess
from models.train import train

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

@router.get("/train")
async def train_model():
    # Add your model training logic here
    train()
    return {"message": "Model training endpoint"}

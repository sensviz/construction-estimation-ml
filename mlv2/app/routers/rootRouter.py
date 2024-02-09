from fastapi import APIRouter
import os
from app.helpers.preprocess import process


router = APIRouter()


# @router.post("/preprocess")
# def question(data):    
#     df = preprocess
# @router.post("/grant")
# async def grant_matching(data: dict):

@router.post("/predict")
async def preprocess_data(data: dict):
    # Add your data preprocessing logic here
    print(data)
    data1 = data['data']
    date = data['currentdate']
    print(data1)
    print(date)
    price = process(data1 , date)
    return price


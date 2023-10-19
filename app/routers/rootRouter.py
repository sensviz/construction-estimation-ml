from fastapi import APIRouter
import os
from helpers.preprocess import preprocess

router = APIRouter(

)


@router.post("/preprocess")
def question(data):
    
    df = preprocess
# @router.post("/grant")
# async def grant_matching(data: dict):

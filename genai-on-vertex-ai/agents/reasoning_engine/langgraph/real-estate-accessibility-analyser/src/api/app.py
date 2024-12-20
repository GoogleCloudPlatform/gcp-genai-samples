from fastapi import FastAPI, Query, HTTPException
from google.cloud import firestore
from pydantic import BaseModel
from typing import Optional, List
from src.config import PROJECT_ID, FIRESTORE_COLLECTION

db = firestore.Client(project=PROJECT_ID)
app = FastAPI()

class House(BaseModel):
    id: str
    title: str
    description: str
    image_urls: List[str]
    is_accessible: bool

@app.get("/houses")
async def list_houses(
    accessible: Optional[bool] = Query(None),
    page: int = Query(1),
    page_size: int = Query(10)
):
    query = db.collection(FIRESTORE_COLLECTION)
    if accessible is not None:
        query = query.where('is_accessible', '==', accessible)
    
    start = (page - 1) * page_size
    docs = query.limit(page_size).offset(start).stream()

    houses = []
    for doc in docs:
        house_data = doc.to_dict()
        house_data['id'] = doc.id
        houses.append(House(**house_data))
    
    return {
        "page": page,
        "page_size": page_size,
        "houses": houses
    }

@app.get("/houses/{house_id}")
async def get_house(house_id: str):
    doc_ref = db.collection(FIRESTORE_COLLECTION).document(house_id)
    doc = doc_ref.get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="House not found")
    house_data = doc.to_dict()
    house_data['id'] = doc.id
    return House(**house_data)

@app.get("/houses/search")
async def search_houses(
    query: str = Query(...),
    page: int = Query(1),
    page_size: int = Query(10)
):
    results = []
    docs = db.collection(FIRESTORE_COLLECTION).stream()
    for doc in docs:
        data = doc.to_dict()
        if (query.lower() in data['title'].lower() or 
            query.lower() in data['description'].lower()):
            data['id'] = doc.id
            results.append(House(**data))
    start = (page - 1) * page_size
    end = start + page_size
    return {
        "page": page,
        "page_size": page_size,
        "houses": results[start:end]
    }

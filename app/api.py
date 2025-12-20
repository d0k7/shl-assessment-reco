# app/api.py
from fastapi import FastAPI
from pydantic import BaseModel
from app.index.search_index import recommend  # Assuming the recommend function is defined here

# Define the request body model
class QueryRequest(BaseModel):
    query: str
    top_k: int = 10

# Initialize FastAPI app
app = FastAPI()

# Define a route for recommending assessments
@app.post("/recommend")
async def recommend_assessment(request: QueryRequest):
    recommendations = recommend(request.query, top_k=request.top_k)
    return {"recommendations": recommendations}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

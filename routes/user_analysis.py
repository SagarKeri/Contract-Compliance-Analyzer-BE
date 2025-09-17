from fastapi import APIRouter, HTTPException

#Import config
from config import db

router = APIRouter()

@router.get("/users/{user_id}/analysis", tags=["User Analysis"])
async def get_user_analysis(user_id: str):
    """
    Fetch all past analysis records for a given user_id.
    """
    try:
        # Validate ObjectId format (in case user_id is Mongo _id)
        # but since you are storing user_id as string, we can just use it directly
        records = list(db.user_analysis.find({"user_id": user_id}, {"_id": 0}))
        
        if not records:
            return {"message": "No analysis records found for this user", "data": []}

        return {"user_id": user_id, "data": records}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching analysis: {str(e)}")


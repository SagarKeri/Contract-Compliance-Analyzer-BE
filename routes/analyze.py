from fastapi import File, UploadFile, Form, APIRouter, HTTPException
from datetime import datetime
from pathlib import Path

#Import Models
from models import FeedbackInput

#Import Helper Functions
from utils.analyze import analyze_contract, compute_cache_key, extract_text_from_pdf

#Import Config
from config import db

router = APIRouter()

UPLOAD_DIR = Path(__file__).resolve().parent / "uploaded_contracts"
UPLOAD_DIR.mkdir(exist_ok=True)

@router.post("/analyze-contract", tags=["Contract Compliance"])
async def analyze_contract_api(
    file: UploadFile = File(...),
    model: str = Form(...),
    clauses: str = Form(...),
    user_id: str = Form(...),
    country_id: str = Form(...),
    domain_id: str = Form(...)
    ):
    start_time = datetime.utcnow()
    analysis_doc_id = None

    try:
        clause_ids_list = sorted([int(c.strip()) for c in clauses.strip("[]").split(",") if c.strip()])
        file_bytes = await file.read()

        # ✅ Compute cache key first
        pdf_hash = compute_cache_key(file_bytes, clause_ids_list, user_id)

        # ✅ Save uploaded PDF in folder
        contract_dir = UPLOAD_DIR / pdf_hash
        contract_dir.mkdir(parents=True, exist_ok=True)

        pdf_path = contract_dir / file.filename
        with open(pdf_path, "wb") as f:
            f.write(file_bytes)

        # ✅ Check cache first (no new log if cached result is valid)
        cached_doc = db.contracts_cache.find_one({"_id": pdf_hash, "user_id": user_id})
        if cached_doc and cached_doc.get("feedback", "").lower() != "dislike":
            return {
                "analysis": cached_doc.get("analysis", []),
                "cachekey": pdf_hash,
                "cached": True,
                "saved_pdf": str(pdf_path)
            }

        # ✅ Insert initial log
        analysis_doc_id = db.user_analysis.insert_one({
            "user_id": user_id,
            "cached_response_id": pdf_hash,
            "pdfName": file.filename,
            "start_time": start_time,
            "end_time": None,
            "is_success": False
        }).inserted_id

        # Extract text and analyze
        text = extract_text_from_pdf(file_bytes)
        result = analyze_contract(text, model, clause_ids_list)

        # Save new record
        db.contracts_cache.update_one(
            {"_id": pdf_hash},
            {
                "$set": {
                    "user_id": user_id,
                    "pdf_name": file.filename,
                    "analysis": result,
                    "clauses": clause_ids_list,
                    "feedback": "",
                    "feedback_given": False,
                    "uploaded_at": datetime.utcnow().isoformat(),
                    "country_id": country_id,
                    "domain_id": domain_id
                }
            },
            upsert=True
        )

        db.user_analysis.update_one(
            {"_id": analysis_doc_id},
            {"$set": {"end_time": datetime.utcnow(), "is_success": True}}
        )

        return {
            "analysis": result,
            "cachekey": pdf_hash,
            "cached": False,
            "saved_pdf": str(pdf_path)  # ✅ return saved path
        }

    except Exception as e:
        if analysis_doc_id:
            db.user_analysis.update_one(
                {"_id": analysis_doc_id},
                {"$set": {"end_time": datetime.utcnow(), "is_success": False}}
            )

        return {
            "error": str(e),
            "cached": False
        }

@router.post("/feedback", tags=["Contract Compliance"])
def submit_feedback(feedback_input: FeedbackInput):
    # Find the contract by cache_key
    contract = db.contracts_cache.find_one({"_id": feedback_input.cache_key})
    if not contract:
        raise HTTPException(status_code=404, detail="Cache key not found")

    # Update the feedback and set feedback_given = True
    db.contracts_cache.update_one(
        {"_id": feedback_input.cache_key},
        {"$set": {
            "feedback": feedback_input.feedback,
            "feedback_given": True
        }}
    )

    return {"message": "Feedback submitted successfully"}

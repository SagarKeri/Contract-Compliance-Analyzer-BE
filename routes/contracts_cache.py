from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse, FileResponse
import os
from pymongo import MongoClient
from bson import ObjectId
import pandas as pd
import io
from pathlib import Path
from openpyxl.utils import get_column_letter
from openpyxl.styles import Font, PatternFill

# Import Helper Functions
from utils.contracts_cache import format_sheet

# Import Config
from config import db

router = APIRouter()

UPLOAD_DIR = Path(__file__).resolve().parent / "uploaded_contracts"
UPLOAD_DIR.mkdir(exist_ok=True)

@router.get("/contracts-cache/{cache_id}", tags=["Contracts Cache"])
async def get_contract_cache(cache_id: str):
    """
    Fetch a cached contract analysis by its ID.
    """
    try:
        record = db.contracts_cache.find_one({"_id": cache_id}, {"_id": 0})
        if not record:
            raise HTTPException(status_code=404, detail="Contract cache not found")

        return {"cache_id": cache_id, "data": record}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching contract cache: {str(e)}")

#--------------------------------download-contract-------------------------------------
@router.get("/download-contract/{cache_id}", tags=["Contracts-Cache"])
async def download_contract(cache_id: str):
    """
    Download the first uploaded contract PDF from uploaded_files/{cache_id}.
    """
    try:
        contract_dir = UPLOAD_DIR / cache_id
        print(f"Looking in: {contract_dir}")  # DEBUG

        if not contract_dir.exists() or not contract_dir.is_dir():
            raise HTTPException(status_code=404, detail="No folder found for this cache_id")

        # Get list of files in the folder
        files = list(contract_dir.glob("*"))
        print(f"Files found: {files}")  # DEBUG

        if not files:
            raise HTTPException(status_code=404, detail="No files found in this cache_id folder")

        # Pick the first file (could sort if you want deterministic order)
        pdf_path = files[0]
        print(f"Returning file: {pdf_path}")  # DEBUG
        print("--------------------------------------------------->" + os.path.basename(str(pdf_path)))

        return FileResponse(
            pdf_path,
            media_type="routerlication/pdf",
            filename=os.path.basename(str(pdf_path))   # ✅ sends original filename
        )

    except Exception as e:
        import traceback
        traceback.print_exc()  # show full error in logs
        raise HTTPException(status_code=500, detail=f"Error fetching PDF: {str(e)}")

@router.get("/download-excel", tags=["Contracts-Cache"])
def download_excel_format(_id: str = Query(...)):
    # Fetch the contract analysis document
    document = db.contracts_cache.find_one({"_id": _id}) or db.contracts_cache.find_one({"_id": ObjectId(_id)})
    if not document:
        raise HTTPException(status_code=404, detail="Analysis not found")

    # Contract metadata
    contract_name = document.get("pdf_name", "Contract.pdf")
    num_clauses = len(document.get("clauses", []))
    description = document.get("analysis", {}).get("description", "")

    # Loop over clauses in the contract
    table_rows = []
    for clause_id in document.get("clauses", []):
        # Fetch clause document
        clause_doc = db.clauses.find_one({"_id": clause_id}) or db.clauses.find_one({"_id": ObjectId(clause_id)})
        clause_name = clause_doc.get("clause_name") if clause_doc else ""
        clause_text = clause_doc.get("clause_text") if clause_doc else ""

        # Match analysis for this clause (safe dict check)
        matched_analysis = next(
            (
                a for a in document.get("analysis", {}).get("analysis", [])
                if isinstance(a, dict) and a.get("compliance_area", "").lower() == clause_name.lower()
            ),
            {}
        )

        table_rows.routerend({
            "Clause": clause_name,
            "Clause Text": clause_text,
            "Status": matched_analysis.get("missing_clause", ""),
            "Reason": matched_analysis.get("reason", ""),
            "Extracted Clause": matched_analysis.get("extracted_text", "")
        })

    # Create DataFrame
    df_table = pd.DataFrame(table_rows, columns=["Clause", "Clause Text", "Status", "Reason", "Extracted Clause"])

    # Write Excel
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # Add top metadata manually
        meta_df = pd.DataFrame({
            0: ["Contract Name", "No of Clauses", "Description"],
            1: [contract_name, num_clauses, description]
        })
        meta_df.to_excel(writer, index=False, header=False, startrow=0)
        # Add table starting from row 4
        df_table.to_excel(writer, index=False, startrow=4)
        format_sheet(writer, "Sheet1")

    output.seek(0)
    filename = f"{contract_name}_report.xlsx"

    return StreamingResponse(
        output,
        media_type="routerlication/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )
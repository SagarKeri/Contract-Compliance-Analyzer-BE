import os
import json
from pathlib import Path
import fitz  # PyMuPDF
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.llms import Ollama
import google.generativeai as genai
import hashlib
from pathlib import Path
import json
from pydantic import BaseModel
from typing import Optional
from pymongo import MongoClient
from io import BytesIO


# ---------- Database Connection ----------
client = MongoClient("mongodb://localhost:27017/")
db = client["Contract-Compliance-Analyzer"]

# ---------- Load Gemini API Key ---------- #
genai.configure(api_key=os.getenv("GEMINI_API_KEY", "AIzaSyAe68plECaeubfxMEOfQolHU8CPx3te4Ss"))

# ---------- Initialize FastAPI App ---------- #
app = FastAPI(
    title="Oil & Gas Contract Compliance API",
    description="Upload oil & gas contracts in PDF format and get compliance analysis across 19 regulatory areas.",
    version="1.6.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#---------------load compliances from the mongo DB-----------#
from typing import List
from fastapi import HTTPException
from pymongo import MongoClient

# Existing MongoDB connection
# client = MongoClient("mongodb://localhost:27017/")
# db = client["Contract-Compliance-Analyzer"]
clauses_collection = db["clauses"] 

from pymongo.errors import PyMongoError

def load_selected_clauses(clause_ids: list[int]) -> list:
    if not clause_ids:
        raise HTTPException(status_code=400, detail="No clause IDs provided.")

    try:
        # Convert to int if stored as int, or to str if stored as string
        clauses = list(
            clauses_collection.find(
                {"_id": {"$in": clause_ids}},  # match your real field type
                {"_id": 0, "clause_name": 1, "clause_text": 1}
            )
        )

        if not isinstance(clauses, list):
            clauses = [clauses]  # ensure always a list

        if not clauses:
            raise HTTPException(status_code=404, detail="No clauses found for given IDs.")

        return clauses

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

# ---------- PDF Text Extractor ---------- #
def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = ""
        for page in doc:
            # Use "text" option to preserve line breaks and whitespace
            page_text = page.get_text("text")
            text += page_text + "\n"  # Add page separator
        return text.strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF processing failed: {str(e)}")

# ---------- Prompt Template ---------- #
def build_prompt(text: str, clauses_text: list, model: str) -> str:
    """
    Build a prompt to check if each clause in clauses_text exists in the contract text.
    clauses_text: list of dicts, each dict has keys 'clause_name' and 'clause_text'
    """
    print(clauses_text)
    return f"""
    You are an expert contract reviewer.

    Analyze the contract text below and check for the presence and adequacy of the following clauses:

    {clauses_text}

    ### Instructions:
    - For each clause:
        - If it is completely missing, return:
            "missing_clause": "Missing: <brief reason>",
            "extracted_text": null
        - If it exists but is vague, generic, or non-compliant, return:
            "missing_clause": "Insufficient: <reason>",
            "extracted_text": "<full original text from contract>"
        - If it exists and is fully adequate, return:
            "missing_clause": "Sufficient",
            "extracted_text": "<full original text from contract>"

    - Preserve exact formatting from the contract in "extracted_text".
    - Output a JSON array of objects with one object per clause:
    [
    {{
        "compliance_area": "<clause_name>",
        "missing_clause": "<Missing / Insufficient / Sufficient>",
        "extracted_text": "<verbatim clause text or null>"
    }},
    ...
    ]

    Return ONLY the JSON array. Do not include any explanation outside the JSON.

    Contract text to review:
    \"\"\"{text}\"\"\"
    """

# ---------- LLM Router ---------- #
def analyze_contract(text: str, model: str, clause_ids: list[int]) -> list[dict]:
    # Load clauses from MongoDB
    clauses = load_selected_clauses(clause_ids)
    print("--------------")
    print(clauses)

    if not isinstance(clauses, list) or not all(isinstance(c, dict) for c in clauses):
        raise HTTPException(status_code=500, detail="Invalid clause format from DB")

    # Build prompt
    clauses_text = "\n\n".join(
        [f"Clause Name: {c['clause_name']}\nClause Text: {c['clause_text']}" for c in clauses]
    )
    prompt = build_prompt(text, clauses_text, model)
    print(prompt)

    if model == "1":
        llm = Ollama(model="mistral", temperature=0)
        return llm.invoke(prompt).strip()
    
    elif model == "2":
        try:
            gemini_model = genai.GenerativeModel("models/gemini-2.5-pro")
            response = gemini_model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Gemini API Error: {str(e)}")

    else:
        raise HTTPException(status_code=400, detail="Unsupported model. Use 'mistral' or 'gemini'.")

client = MongoClient("mongodb://localhost:27017/")
db = client["Contract-Compliance-Analyzer"]
contracts_cache = db["contracts_cache"]

def compute_pdf_hash(file_bytes: bytes) -> str:
    """Generate a unique hash for a PDF file"""
    return hashlib.sha256(file_bytes).hexdigest()

@app.post("/analyze-contract", tags=["Contract Compliance"])
async def analyze_contract_api(
    file: UploadFile = File(...),
    model: str = Form(...),
    clauses: str = Form(...)
    ):
    try:
        print(f"Raw clauses from form: {clauses}")
        clause_ids_list = [int(c.strip()) for c in clauses.strip("[]").split(",") if c.strip()]
        print(f"Parsed clause IDs: {clause_ids_list}")

        file_bytes = await file.read()
        pdf_hash = compute_pdf_hash(file_bytes)

        # Check if cached result exists
        cached_doc = contracts_cache.find_one({"_id": pdf_hash})
        if cached_doc:
            print("Returning cached analysis from MongoDB")
            return JSONResponse(content={"analysis": cached_doc["analysis"]})

        # Extract text and analyze
        text = extract_text_from_pdf(file_bytes)
        raw_result = analyze_contract(text, model, clause_ids_list)

        if isinstance(raw_result, (list, dict)):
            parsed_result = raw_result
        elif isinstance(raw_result, str):
            try:
                parsed_result = json.loads(
                    raw_result.replace("```json", "").replace("```", "").strip()
                )
            except json.JSONDecodeError:
                raise HTTPException(status_code=500, detail="LLM did not return valid JSON.")
        else:
            raise HTTPException(status_code=500, detail="Unexpected result type from analysis.")

        # Store result in MongoDB
        contracts_cache.update_one(
            {"_id": pdf_hash},
            {"$set": {"analysis": parsed_result, "clauses": clause_ids_list}},
            upsert=True
        )

        return JSONResponse(content={"analysis": parsed_result})

    except Exception as e:
        print(f"Error in analyze_contract_api: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in analyze_contract_api: {str(e)}")

# #-----------Compute PDF Hash--------#
# def compute_pdf_hash(file_bytes: bytes) -> str:
#     return hashlib.sha256(file_bytes).hexdigest()

#---------Read Cache File if Exists-----#
CACHE_DIR = Path(__file__).resolve().parent / "json_cache"
CACHE_DIR.mkdir(exist_ok=True)

def load_cached_response(pdf_hash):
    cache_file = f"./cache/{pdf_hash}.json"
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:  # "r" mode for reading
            return json.load(f)
    return None

def save_response_to_cache(pdf_hash: str, data: dict):
    cache_path = CACHE_DIR / f"{pdf_hash}.json"
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

from pydantic import BaseModel
from fastapi import HTTPException

class FeedbackInput(BaseModel):
    compliance_area: str
    feedback: str
    cache_key: str

@app.post("/feedback", tags=["Contract Compliance"])
def submit_feedback(feedback_input: FeedbackInput):
    feedback_file_path = CACHE_DIR / f"{feedback_input.cache_key}.json"

    if not feedback_file_path.exists():
        raise HTTPException(status_code=404, detail="Cache key not found")

    with open(feedback_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if data.get("feedback") is None:
        data["feedback"] = {}
    data["feedback"][feedback_input.compliance_area] = feedback_input.feedback

    with open(feedback_file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return {"message": "Feedback submitted successfully"}

import json
import tempfile
import shutil

def atomic_write_json(data, filepath):
    with tempfile.NamedTemporaryFile("w", delete=False, dir=os.path.dirname(filepath)) as tmpfile:
        json.dump(data, tmpfile, indent=2)
        tempname = tmpfile.name
    shutil.move(tempname, filepath)

import hashlib

def generate_cache_key(text: str, country: str) -> str:
    """
    Generate a unique cache key based on PDF text and country.
    """
    combined = f"{text}-{country}"
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()

# # ---------- Feedback Input ---------- #
# class FeedbackInput(BaseModel):
#     compliance_area: str
#     feedback: str
#     cache_key: str
# # ---------- Save PDF Copy ---------- #
# @app.post("/save-copy", tags=["PDF Management"])
# async def save_pdf_copy(file: UploadFile = File(...)):
#     if file.content_type != "application/pdf":
#         raise HTTPException(status_code=400, detail="Only PDF files are supported.")

#     contents = await file.read()
#     copy_dir = Path(__file__).resolve().parent.parent / "copypdf"
#     copy_dir.mkdir(exist_ok=True)
#     save_path = copy_dir / "uploaded_contract.pdf"

#     with open(save_path, "wb") as f:
#         f.write(contents)

#     return {"message": "PDF saved successfully.", "file_path": str(save_path)}

# # ---------- Download PDF ---------- #
# @app.get("/download-pdf", tags=["PDF Management"])
# async def download_saved_pdf():
#     pdf_path = Path(__file__).resolve().parent.parent / "copypdf" / "uploaded_contract.pdf"
#     if not pdf_path.exists():
#         raise HTTPException(status_code=404, detail="PDF file not found.")
#     return FileResponse(path=str(pdf_path), media_type="application/pdf", filename="contract.pdf")


# ---------- Country JSON Mapping ---------- #
# COUNTRY_JSON_MAP = {
#     "1": "USA.json",
#     "2": "UK.json",
#     "3": "IND.json",
#     "4": "UAE.json",
#     "5": "Saudi_Arabia.json",
#     "6": "Qatar.json"
# }

# ---------- Load Country Laws ---------- #
# def load_country_laws(country_id: str) -> list:
#     laws_dir = Path(__file__).resolve().parent / "laws"
#     json_file = COUNTRY_JSON_MAP.get(country_id)
    
#     if not json_file:
#         raise HTTPException(status_code=400, detail=f"Invalid country_id: {country_id}. Supported IDs: {list(COUNTRY_JSON_MAP.keys())}")
    
#     json_path = laws_dir / json_file
#     if not json_path.exists():
#         raise HTTPException(status_code=404, detail=f"Law file for country_id {country_id} not found.")
    
#     try:
#         with open(json_path, "r") as f:
#             return json.load(f)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to load laws for country_id {country_id}: {str(e)}")



from fastapi import FastAPI
from pydantic import BaseModel
from bson import ObjectId


class Country(BaseModel):
    country_name: str

class Domain(BaseModel):
    domain_name: str
    country_id: int

class Compliance(BaseModel):
    compliance_name: str
    domain_id: int

class Clause(BaseModel):
    clause_name:str
    clause_text: str
    domain_id: Optional[int] = None

# ---------- Auto Increment Function ----------
def get_next_sequence(name):
    counter = db.counters.find_one_and_update(
        {"_id": name},
        {"$inc": {"sequence_value": 1}},
        upsert=True,
        return_document=True
    )
    return counter["sequence_value"]


from fastapi import HTTPException

# ---------- CRUD for Country ----------
@app.post("/countries",tags=["Countries"])
def create_country(country: Country):
    next_id = get_next_sequence("country_id")
    db.countries.insert_one({"_id": next_id, "country_name": country.country_name})
    return {"message": "Country added", "id": next_id}

@app.get("/countries",tags=["Countries"])
def get_countries():
    return list(db.countries.find({}, {"_id": 1, "country_name": 1}))

@app.put("/countries/{country_id}",tags=["Countries"])
def update_country(country_id: int, country: Country):
    result = db.countries.update_one({"_id": country_id}, {"$set": {"country_name": country.country_name}})
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Country not found")
    return {"message": "Country updated"}

@app.delete("/countries/{country_id}",tags=["Countries"])
def delete_country(country_id: int):
    result = db.countries.delete_one({"_id": country_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Country not found")
    return {"message": "Country deleted"}

@app.get("/countries/{country_id}", tags=["Countries"])
def get_country_by_id(country_id: int):
    country = db.countries.find_one({"_id": country_id}, {"_id": 1, "country_name": 1})
    if not country:
        raise HTTPException(status_code=404, detail="Country not found")
    return country


# ===== DOMAIN CRUD =====
@app.post("/domains",tags=["Domains"])
def create_domain(domain: Domain):
    next_id = get_next_sequence("domain_id")
    db.domains.insert_one({
        "_id": next_id,
        "domain_name": domain.domain_name,
        "country_id": domain.country_id
    })
    return {"message": "Domain added", "id": next_id}

@app.get("/domains", tags=["Domains"])
def get_domains():
    pipeline = [
        {
            "$lookup": {
                "from": "countries",        # Join with countries collection
                "localField": "country_id", # Field in domains
                "foreignField": "_id",      # Field in countries
                "as": "country_info"
            }
        },
        {"$unwind": "$country_info"},       # Flatten the array
        {
            "$project": {                    # Select fields to return
                "_id": 1,
                "domain_name": 1,
                "country_id": 1,
                "country_name": "$country_info.country_name"
            }
        }
    ]
    
    domains = list(db.domains.aggregate(pipeline))
    return domains 


@app.get("/domains/byid/{domain_id}", tags=["Domains"])
def get_domain_by_id(domain_id: int):
    pipeline = [
        {"$match": {"_id": domain_id}},
        {
            "$lookup": {
                "from": "countries",
                "localField": "country_id",
                "foreignField": "_id",
                "as": "country_info"
            }
        },
        {"$unwind": "$country_info"},
        {
            "$project": {
                "_id": 1,
                "domain_name": 1,
                "country_id": 1,
                "country_name": "$country_info.country_name"
            }
        }
    ]
    
    result = list(db.domains.aggregate(pipeline))
    if not result:
        raise HTTPException(status_code=404, detail="Domain not found")
    return result[0]

@app.put("/domains/{domain_id}",tags=["Domains"])
def update_domain(domain_id: int, domain: Domain):
    result = db.domains.update_one({"_id": domain_id}, {"$set": {
        "domain_name": domain.domain_name,
        "country_id": domain.country_id
    }})
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Domain not found")
    return {"message": "Domain updated"}

@app.delete("/domains/{domain_id}",tags=["Domains"])
def delete_domain(domain_id: int):
    result = db.domains.delete_one({"_id": domain_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Domain not found")
    return {"message": "Domain deleted"}

@app.get("/domains/bycountry/{country_id}", tags=["Domains"])
def get_domains_by_country(country_id: int):
    pipeline = [
        {"$match": {"country_id": country_id}},
        {
            "$lookup": {
                "from": "countries",        # Join with countries collection
                "localField": "country_id", # Field in domains
                "foreignField": "_id",      # Field in countries
                "as": "country_info"
            }
        },
        {"$unwind": "$country_info"},       # Flatten array
        {
            "$project": {                    # Fields to return
                "_id": 1,
                "domain_name": 1,
                "country_id": 1,
                "country_name": "$country_info.country_name"
            }
        }
    ]

    domains = list(db.domains.aggregate(pipeline))
    return domains

# ===== COMPLIANCE CRUD =====
@app.get("/compliances", tags=["Compliances"])
def get_all_compliances():
    pipeline = [
        {
            "$lookup": {
                "from": "domains",        # Join with domains collection
                "localField": "domain_id", # Field in compliances
                "foreignField": "_id",     # Field in domains
                "as": "domain_info"
            }
        },
        {"$unwind": "$domain_info"},    # Flatten array to object
        {
            "$project": {               # Select fields to return
                "_id": 1,
                "compliance_name": 1,
                "domain_id": 1,
                "domain_name": "$domain_info.domain_name"
            }
        }
    ]
    
    compliances = list(db.compliances.aggregate(pipeline))
    return compliances

@app.post("/compliances",tags=["Compliances"])
def create_compliance(compliance: Compliance):
    next_id = get_next_sequence("compliance_id")
    db.compliances.insert_one({
        "_id": next_id,
        "compliance_name": compliance.compliance_name,
        "domain_id": compliance.domain_id
    })
    return {"message": "Compliance added", "id": next_id}

@app.get("/compliances/{domain_id}", tags=["Compliances"])
def get_compliances(domain_id: int):
    pipeline = [
        {"$match": {"domain_id": domain_id}},  # Filter by domain_id
        {
            "$lookup": {
                "from": "domains",          # Join with domains collection
                "localField": "domain_id",  # Field in compliances
                "foreignField": "_id",      # Field in domains
                "as": "domain_info"
            }
        },
        {"$unwind": "$domain_info"},            # Flatten array
        {
            "$project": {
                "_id": 1,
                "compliance_name": 1,
                "domain_id": 1,
                "domain_name": "$domain_info.domain_name"
            }
        }
    ]

    compliances = list(db.compliances.aggregate(pipeline))
    return compliances

@app.put("/compliances/{compliance_id}",tags=["Compliances"])
def update_compliance(compliance_id: int, compliance: Compliance):
    result = db.compliances.update_one({"_id": compliance_id}, {"$set": {
        "compliance_name": compliance.compliance_name,
        "domain_id": compliance.domain_id
    }})
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Compliance not found")
    return {"message": "Compliance updated"}

@app.delete("/compliances/{compliance_id}",tags=["Compliances"])
def delete_compliance(compliance_id: int):
    result = db.compliances.delete_one({"_id": compliance_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Compliance not found")
    return {"message": "Compliance deleted"}

# ===== CLAUSE CRUD =====
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

class Clause(BaseModel):
    clause_name: str
    clause_text: str
    domain_id: Optional[int] = None

# ===== CLAUSE CRUD =====

# Create Clause
@app.post("/clauses", tags=["Clauses"])
def create_clause(clause: Clause):
    next_id = get_next_sequence("clause_id")  # Your ID generator
    db.clauses.insert_one({
        "_id": next_id,
        "clause_name": clause.clause_name,
        "clause_text": clause.clause_text,
        "domain_id": clause.domain_id
    })
    return {"message": "Clause added", "id": next_id}

# Get all clauses with domain info
@app.get("/clauses", tags=["Clauses"])
def get_all_clauses():
    pipeline = [
                    {
                        "$lookup": {
                            "from": "domains",
                            "localField": "domain_id",
                            "foreignField": "_id",
                            "as": "domain_info"
                        }
                    },
                    {"$unwind": {"path": "$domain_info", "preserveNullAndEmptyArrays": True}},
                    {
                        "$project": {
                            "_id": 1,
                            "clause_name": 1,
                            "clause_text": 1,
                            "domain_id": 1,
                            "domain_name": "$domain_info.domain_name"
                        }
                    }
    ]
    clauses = list(db.clauses.aggregate(pipeline))
    return clauses

# Get clauses by country & domain (filter)
@app.get("/clauses/filter", tags=["Clauses"])
def get_clauses_by_country_domain(
    country_id: int = Query(..., description="Country ID"),
    domain_id: int = Query(..., description="Domain ID")
    ):
    pipeline = [
        {"$match": {"domain_id": domain_id}},
        {
            "$lookup": {
                "from": "domains",
                "localField": "domain_id",
                "foreignField": "_id",
                "as": "domain_info"
            }
        },
        {"$unwind": "$domain_info"},
        {"$match": {"domain_info.country_id": country_id}},
        {
            "$project": {
                "_id": 1,
                "clause_name": 1,
                "clause_text": 1,
                "domain_id": 1,
                "domain_name": "$domain_info.domain_name",
                "country_id": "$domain_info.country_id"
            }
        }
    ]
    clauses = list(db.clauses.aggregate(pipeline))
    return clauses

# Get clauses by domain (single path parameter)
@app.get("/clauses/by-domain/{domain_id}", tags=["Clauses"])
def get_clauses_by_domain(domain_id: int):
    clauses = list(db.clauses.find(
        {"domain_id": domain_id},
        {"_id": 1, "clause_name": 1, "clause_text": 1, "domain_id": 1}
    ))
    return clauses

# Update Clause
@app.put("/clauses/{clause_id}", tags=["Clauses"])
def update_clause(clause_id: int, clause: Clause):
    result = db.clauses.update_one({"_id": clause_id}, {"$set": {
        "clause_name": clause.clause_name,
        "clause_text": clause.clause_text,
        "domain_id": clause.domain_id
    }})
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Clause not found")
    return {"message": "Clause updated"}

# Delete Clause
@app.delete("/clauses/{clause_id}", tags=["Clauses"])
def delete_clause(clause_id: int):
    result = db.clauses.delete_one({"_id": clause_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Clause not found")
    return {"message": "Clause deleted"}
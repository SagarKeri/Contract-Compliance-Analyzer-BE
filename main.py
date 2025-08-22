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
genai.configure(api_key=os.getenv("GEMINI_API_KEY", "AIzaSyDQs8IFeuZvCd_erNBrEXU2Q8rqaCUG-pc"))

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
    Build a prompt to:
    1. Generate a brief description of the uploaded contract PDF.
    2. Check if each clause in clauses_text exists in the contract text.
       clauses_text: list of dicts, each dict has keys 'clause_name' and 'clause_text'
    """
    print(clauses_text)
    return f"""
    You are an expert contract reviewer.

    Task 1: Provide a brief description of the uploaded contract text. 
    - Summarize in 2–3 sentences what the contract is about, covering key parties, purpose, and type of obligations.

    Task 2: Analyze the contract text and check for the presence and adequacy of the following clauses:

    {clauses_text}

    ### Instructions:
    - For each clause:
        - If it is completely missing, return:
            "missing_clause": "Missing",
            "reason":"<brief reason>",
            "extracted_text": null
        - If it exists but is vague, generic, or non-compliant, return:
            "missing_clause": "Insufficient",
            "reason":"<brief reason>",
            "extracted_text": "<first 50 words from the original clause in the contract>"
        - If it exists and is fully adequate, return:
            "missing_clause": "Sufficient",
            "reason":"<brief reason>",
            "extracted_text": "<first 50 words from the original clause in the contract>"

    - Preserve exact formatting from the contract in "extracted_text".
    - Output a single JSON object in the format:

    {{
      "description": "<2–3 sentence summary of the contract>",
      "analysis": [
        {{
          "compliance_area": "<clause_name>",
          "missing_clause": "<Missing / Insufficient / Sufficient>",
          "reason": "<brief reason>",
          "extracted_text": "<verbatim clause text or null>"
        }},
        ...
      ]
    }}

    Return ONLY the JSON object. Do not include any explanation outside the JSON.

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

        # ✅ Check cache
        cached_doc = contracts_cache.find_one({"_id": pdf_hash})
        if cached_doc:
            # If feedback was "dislike", re-run LLM
            if cached_doc.get("feedback", "").lower() == "dislike":
                print("Feedback = dislike → re-running LLM")
            else:
                print("Returning cached analysis from MongoDB")
                return {
                    "analysis": cached_doc.get("analysis", []),
                    "cachekey": pdf_hash
                }

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
        print(pdf_hash)
        # Save to MongoDB
        contracts_cache.update_one(
            {"_id": pdf_hash},
            {
                "$set": {
                    "analysis": parsed_result,
                    "clauses": clause_ids_list,
                    "feedback": "",
                    "feedback_given": False
                }
            },
            upsert=True
        )

        return {
            "analysis": parsed_result,
            "cachekey": pdf_hash
        }

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
    feedback: str
    cache_key: str   # _id of the document

@app.post("/feedback", tags=["Contract Compliance"])
def submit_feedback(feedback_input: FeedbackInput):
    # Find the contract by cache_key
    contract = contracts_cache.find_one({"_id": feedback_input.cache_key})
    if not contract:
        raise HTTPException(status_code=404, detail="Cache key not found")

    # Update the feedback and set feedback_given = True
    contracts_cache.update_one(
        {"_id": feedback_input.cache_key},
        {"$set": {
            "feedback": feedback_input.feedback,
            "feedback_given": True
        }}
    )

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

# Get Clause by ID
@app.get("/clauses/{clause_id}", tags=["Clauses"])
def get_clause_by_id(clause_id: int):
    clause = db.clauses.find_one(
        {"_id": clause_id},
        {"_id": 1, "clause_name": 1, "clause_text": 1, "domain_id": 1}
    )

    if not clause:
        raise HTTPException(status_code=404, detail="Clause not found")

    # Fetch domain details
    domain = db.domains.find_one(
        {"_id": clause["domain_id"]},
        {"_id": 1, "domain_name": 1}
    )

    # Add domain_name if found
    clause["domain_name"] = domain["domain_name"] if domain else None

    return clause

# Delete Clause
@app.delete("/clauses/{clause_id}", tags=["Clauses"])
def delete_clause(clause_id: int):
    result = db.clauses.delete_one({"_id": clause_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Clause not found")
    return {"message": "Clause deleted"}


#--------------------------------------------chat Bot-------------------------------------------------
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from fastapi import UploadFile, File, Form, HTTPException
from pydantic import BaseModel
import json
import re
from ollama import Client  # ✅ Uncomment and fix
import google.generativeai as genai
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# In-memory FAISS cache
vector_store_cache = {}

class ChatbotFileInput(BaseModel):
    question: str
    model: str  # "1" = Mistral, "2" = Gemini

def is_greeting(question: str) -> bool:
    """Check if the question is a greeting"""
    greetings = [
        r'\b(hi|hello|hey|good morning|good afternoon|good evening)\b',
        r'^\s*(hi|hello|hey)\s*[!.?]*\s*$'
    ]
    question_lower = question.lower().strip()
    return any(re.search(pattern, question_lower, re.IGNORECASE) for pattern in greetings)

def is_contract_related(question: str, context: str = "") -> bool:
    """Check if the question is related to contracts or legal documents"""
    contract_keywords = [
        'contract', 'agreement', 'clause', 'term', 'condition', 'obligation', 
        'liability', 'payment', 'delivery', 'breach', 'termination', 'penalty',
        'warranty', 'indemnity', 'confidentiality', 'non-disclosure', 'nda',
        'jurisdiction', 'governing law', 'dispute', 'arbitration', 'renewal',
        'amendment', 'modification', 'force majeure', 'intellectual property',
        'proprietary', 'damages', 'compensation', 'fee', 'cost', 'price',
        'schedule', 'milestone', 'deliverable', 'specification', 'requirement',
        'compliance', 'regulatory', 'legal', 'law', 'rights', 'duties'
    ]
    
    question_lower = question.lower()
    
    # Check if question contains contract-related keywords
    has_contract_keywords = any(keyword in question_lower for keyword in contract_keywords)
    
    # Check if context (retrieved from PDF) contains relevant information
    has_relevant_context = bool(context and context.strip())
    
    # If we have relevant context from the contract PDF, it's likely contract-related
    if has_relevant_context:
        return True
    
    # If question has contract keywords, it's likely contract-related
    if has_contract_keywords:
        return True
    
    # Check for document-specific questions
    document_questions = [
        'what does this say', 'what is in this', 'summarize this', 'explain this',
        'what are the', 'tell me about', 'find', 'search', 'look for'
    ]
    
    if any(phrase in question_lower for phrase in document_questions):
        return True
    
    return False

@app.post("/chatbot-file", tags=["Contract Chatbot"])
async def chatbot_file(
    file: UploadFile = File(...),
    question: str = Form(...),
    model: str = Form(...)
    ):
    try:
        # Handle greetings first
        if is_greeting(question):
            return {
                "answer": "Hi, I am Contract Genie. How can I help you?",
                "cachekey": "greeting"
            }

        # Step 1: Read PDF
        file_bytes = await file.read()
        pdf_hash = compute_pdf_hash(file_bytes)

        # Step 2: Check MongoDB cache
        cache_key = f"{pdf_hash}_{question}_{model}"
        cached_response = db.contracts_cache.find_one({"_id": cache_key})
        if cached_response and cached_response.get("feedback", "").lower() != "dislike":
            logger.debug(f"Returning cached response for {cache_key}")
            return {
                "answer": cached_response["answer"],
                "cachekey": pdf_hash
            }

        # Step 3: Check vector cache and retrieve context
        if pdf_hash not in vector_store_cache:
            text = extract_text_from_pdf(file_bytes)
            if not text.strip():
                raise HTTPException(status_code=400, detail="No text extracted from PDF")

            # Split into smaller chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            chunks = text_splitter.split_text(text)

            # Build FAISS vector store
            vector_store = FAISS.from_texts(chunks, embedding_model)
            vector_store_cache[pdf_hash] = vector_store
        else:
            vector_store = vector_store_cache[pdf_hash]

        # Step 4: Retrieve relevant chunks
        retriever = vector_store.as_retriever(search_kwargs={"k": 2})
        docs = retriever.invoke(question)
        context = "\n\n".join([d.page_content for d in docs])
        logger.debug(f"Context: {context}")
        logger.debug(f"Question: {question}")

        # Step 5: Check if question is contract-related
        if not is_contract_related(question, context):
            return {
                "answer": "Can't answer this query.",
                "cachekey": pdf_hash
            }

        # Step 6: Build prompt
        prompt = f"""
        You are an expert contract assistant. Answer the question based on the provided contract excerpts.

        Context:
        {context}

        Question:
        {question}

        Instructions:
        - Answer in plain text, clear and concise.
        - Do not include JSON, markdown, or extra explanations.
        - If no relevant information is found, say "Sorry I can't answer this query, try another query."
        - Only answer questions related to contracts, legal documents, or the provided document.

        Answer:
        """

        # Step 7: Call LLM
        answer = None
        if model == "1":
            client = Client(host="http://localhost:11434")
            response = client.chat(
                model="mistral",
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0}
            )
            answer = response["message"]["content"].strip()

        elif model == "2":
            try:
                gemini_model = genai.GenerativeModel("models/gemini-2.5-pro")
                response = gemini_model.generate_content(prompt)
                logger.debug(f"Gemini raw response: {json.dumps(response.to_dict(), indent=2)}")
                if hasattr(response, "candidates") and response.candidates:
                    for candidate in response.candidates:
                        if hasattr(candidate.content, "parts") and candidate.content.parts:
                            answer = candidate.content.parts[0].text.strip()
                            break
                if not answer and hasattr(response, "text") and response.text:
                    answer = response.text.strip()
                if not answer:
                    logger.error(f"No valid content in Gemini response: {response.to_dict()}")
                    raise ValueError("No valid content in Gemini response")
                logger.debug(f"Token usage: {response.usage_metadata.prompt_token_count} prompt, {response.usage_metadata.total_token_count} total")
            except Exception as e:
                logger.error(f"Gemini API error: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Gemini API error: {str(e)}")

        else:
            raise HTTPException(status_code=400, detail="Invalid model. Use '1' or '2'.")

        # Step 8: Cache response
        db.contracts_cache.update_one(
            {"_id": cache_key},
            {
                "$set": {
                    "answer": answer or "No information found",
                    "model": model,
                    "question": question,
                    "feedback": "",
                    "feedback_given": False
                }
            },
            upsert=True
        )

        return {
            "answer": answer or "No information found",
            "cachekey": pdf_hash
        }

    except Exception as e:
        logger.error(f"Chatbot error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chatbot error: {str(e)}")


# Optional: Add a chat-only endpoint for questions without file upload
@app.post("/chatbot-chat", tags=["Contract Chatbot"])
async def chatbot_chat(
    question: str = Form(...),
    model: str = Form(...)
    ):
    """Handle chat without file upload - only for greetings and general contract questions"""
    try:
        # Handle greetings
        if is_greeting(question):
            return {
                "answer": "Hi, I am Contract Genie. How can I help you?",
                "cachekey": "greeting"
            }
        
        # For non-greeting questions without a file, redirect to file upload
        return {
            "answer": "Please upload a contract document so I can help you with your question.",
            "cachekey": "no_file"
        }
    
    except Exception as e:
        logger.error(f"Chat error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")
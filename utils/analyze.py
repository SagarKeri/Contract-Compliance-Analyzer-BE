import hashlib
import fitz 
import json
import re
import os
#import tempfile
#import shutil
#from pathlib import Path
from langchain_community.llms import Ollama
import google.generativeai as genai
from fastapi import HTTPException
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

#Config
from config import db, embedding_model, query_endpoint

def load_selected_clauses(clause_ids: list[int]) -> list:
    if not clause_ids:
        raise HTTPException(status_code=400, detail="No clause IDs provided.")

    try:
        # Convert to int if stored as int, or to str if stored as string
        clauses = list(
            db.clauses.find(
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

# ---------- Prompt Template ---------- #
def build_prompt(summary_context: str, clauses_section: str, model: str) -> str:
    """
    Build a prompt to:
    1. Generate a brief description of the uploaded contract PDF based on summary context.
    2. Check if each clause exists in the contract using provided relevant excerpts.
    """
    return f"""
    You are an expert contract reviewer.

    Task 1: Provide a brief description of the uploaded contract based on the following excerpt: 
    {summary_context}
    - The description MUST be at least 50 words long.
    - It should be written in 2–3 complete sentences.
    - Cover key parties, purpose, subject matter, and type of obligations.

    Task 2: Analyze the contract and check for the presence and adequacy of the following clauses using their respective relevant excerpts:

    {clauses_section}

    ### Instructions:
    - For each clause:
        - Use only the provided relevant excerpts for that clause.
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
          "extracted_text": "<first 50 words from the original clause in the contract>"
        }},
        ...
      ]
    }}

    Return ONLY the JSON object. Do not include any explanation outside the JSON.
    """

def build_summary_prompt(context: str) -> str:
    """
    Build a prompt to generate a brief summary of the contract.
    """
    return f"""
    You are an expert contract reviewer. Based on the following contract excerpt, provide a brief summary of the contract in 2–3 sentences (at least 50 words). Include key parties, purpose, subject matter, and type of obligations.

    Contract Excerpt:
    {context}

    Instructions:
    - Return only the summary as plain text.
    - Ensure the summary is clear, concise, and meets the word count requirement.
    """

def safe_json_loads(raw_result: str) -> dict:
    """
    Clean LLM output and parse as JSON safely.
    """
    if not raw_result:
        raise ValueError("Empty LLM response")

    # Remove code fences if any
    cleaned = raw_result.replace("```json", "").replace("```", "").strip()

    # Extract only the JSON part between the first { and the last }
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError(f"No valid JSON object found in response: {cleaned[:200]}")
    
    json_str = match.group(0)

    return json.loads(json_str)

def analyze_contract(text: str, model: str, clause_ids: list[int]) -> dict:
    # Load clauses
    clauses = load_selected_clauses(clause_ids)
    if not isinstance(clauses, list) or not all(isinstance(c, dict) for c in clauses):
        raise HTTPException(status_code=500, detail="Invalid clause format from DB")

    # Init embeddings + vector store
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    vector_store = FAISS.from_texts(chunks, embedding_model)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # Step 1: Generate summary
    summary_query = "Contract overview: parties involved, main purpose, key obligations"
    summary_docs = retriever.invoke(summary_query)
    summary_context = "\n\n".join([d.page_content for d in summary_docs])
    summary_prompt = build_summary_prompt(summary_context)

    # Call LLM for summary
    summary_result = None
    if model == "1":
        llm = Ollama(model="mistral", temperature=0)
        summary_result = llm.invoke(summary_prompt).strip()
    elif model == "2":
        gemini_model = genai.GenerativeModel("models/gemini-2.5-pro")
        summary_result = gemini_model.generate_content(summary_prompt).text.strip()
    elif model == "3":
        response = query_endpoint(summary_prompt)
        parsed = json.loads(response)
        summary_result = parsed["response"] if "response" in parsed else parsed
    else:
        raise HTTPException(status_code=400, detail="Unsupported model.")

    # Step 2: Analyze clauses (existing logic)
    final_results = []
    batch_size = 3
    for i in range(0, len(clauses), batch_size):
        batch = clauses[i:i + batch_size]
        contexts = {}
        for clause in batch:
            query = f"{clause['clause_name']}: {clause['clause_text']}"
            docs = retriever.invoke(query)
            context = "\n\n".join([d.page_content for d in docs])
            contexts[clause['clause_name']] = context

        clauses_section = "\n\n".join(
            [f"Clause Name: {c['clause_name']}\nRequired Clause Text: {c['clause_text']}\nRelevant Contract Excerpts: {contexts.get(c['clause_name'], '')}" for c in batch]
        )
        prompt = build_prompt(summary_context, clauses_section, model)

        # Call LLM for clause analysis
        raw_result = None
        if model == "1":
            llm = Ollama(model="mistral", temperature=0)
            raw_result = llm.invoke(prompt).strip()
        elif model == "2":
            gemini_model = genai.GenerativeModel("models/gemini-2.5-pro")
            raw_result = gemini_model.generate_content(prompt).text.strip()
        elif model == "3":
            response = query_endpoint(prompt)
            parsed = json.loads(response)
            raw_result = parsed["response"] if "response" in parsed else parsed
        else:
            raise HTTPException(status_code=400, detail="Unsupported model.")

        # Parse JSON safely
        if isinstance(raw_result, str):
            cleaned_result = raw_result.replace("```json", "").replace("```", "").strip()
            if not cleaned_result:
                raise HTTPException(status_code=500, detail="Empty response from LLM")
            try:
                parsed_batch = safe_json_loads(raw_result if isinstance(raw_result, str) else json.dumps(raw_result))
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Invalid JSON from LLM: {str(e)}")

        elif isinstance(raw_result, dict):
            parsed_batch = raw_result
        else:
            raise HTTPException(status_code=500, detail="Unexpected result type.")

        # Merge analysis results
        if "analysis" in parsed_batch:
            final_results.extend(parsed_batch["analysis"])

    # Step 3: Combine summary and analysis
    return {
        "description": summary_result or "Summary could not be generated.",
        "analysis": final_results
    }

def compute_cache_key(file_bytes: bytes, clause_ids: list[int], user_id: str) -> str:
    normalized_clauses = sorted(clause_ids)
    combined = {
        "pdf_hash": hashlib.sha256(file_bytes).hexdigest(),
        "clauses": normalized_clauses,
        "user_id": user_id
    }
    return hashlib.sha256(json.dumps(combined, sort_keys=True).encode()).hexdigest()

def compute_pdf_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()

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


#---------Read Cache File if Exists-----#
#CACHE_DIR = Path(__file__).resolve().parent / "json_cache"
#CACHE_DIR.mkdir(exist_ok=True)
#
#def load_cached_response(pdf_hash):
#    cache_file = f"./cache/{pdf_hash}.json"
#    if os.path.exists(cache_file):
#        with open(cache_file, "r", encoding="utf-8") as f:  # "r" mode for reading
#            return json.load(f)
#    return None
#
#def save_response_to_cache(pdf_hash: str, data: dict):
#    cache_path = CACHE_DIR / f"{pdf_hash}.json"
#    with open(cache_path, "w", encoding="utf-8") as f:
#        json.dump(data, f, indent=2)
#
#def atomic_write_json(data, filepath):
#    with tempfile.NamedTemporaryFile("w", delete=False, dir=os.path.dirname(filepath)) as tmpfile:
#        json.dump(data, tmpfile, indent=2)
#        tempname = tmpfile.name
#    shutil.move(tempname, filepath)
#
#def generate_cache_key(text: str, country: str) -> str:
#    """
#    Generate a unique cache key based on PDF text and country.
#    """
#    combined = f"{text}-{country}"
#    return hashlib.sha256(combined.encode("utf-8")).hexdigest()

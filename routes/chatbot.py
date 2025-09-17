from fastapi import APIRouter
import logging
from fastapi import UploadFile, File, Form, HTTPException
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from fastapi import UploadFile, File, Form, HTTPException

#Import Models
from models import ChatbotMetadataInput, ChatbotFileInput

#Import Helper Functions 
from utils.chatbot import *
from utils.analyze import compute_pdf_hash, extract_text_from_pdf

#Import Config
from config import embedding_model, query_endpoint, db


# Set up logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory FAISS cache
vector_store_cache = {}

@router.post("/chatbot-metadata", tags=["Contract Chatbot"])
async def chatbot_metadata(input_data: ChatbotMetadataInput):
    try:
        question = input_data.question
        model = input_data.model

        # Greeting
        if is_greeting(question):
            return {
                "answer": "Hi, I am Contract Genie. I can help you with information about countries, domains, and clauses in our database. What would you like to know?",
                "cachekey": "greeting_metadata"
            }

        # Cache check
        cache_key = f"metadata_{question}_{model}"
        #cached = db.contracts_cache.find_one({"_id": cache_key})
        #if cached and cached.get("feedback", "").lower() != "dislike":
        #    return {"answer": cached["answer"], "cachekey": "metadata"}
        cached_response = db.contracts_cache.find_one({"_id": cache_key})
        if cached_response and cached_response.get("answer"):
            return {"answer": cached_response["answer"], "cachekey": "metadata"}

        # Detect intent + search
        intent = detect_intent(question)
        print(f"Detected intent: {intent}")
        countries, domains, clauses = search_metadata_entities(question)
        print(f"Found {countries} countries, {domains} domains, {clauses} clauses")

        # Format response
        answer = format_metadata_response(question, intent, countries, domains, clauses)

        # Cache
        if is_valid_answer(answer):
            db.contracts_cache.update_one(
                {"_id": cache_key},
                {"$set": {
                    "answer": answer,
                    "model": model,
                    "question": question,
                    "feedback": "",
                    "feedback_given": False
                }},
                upsert=True
            )

        return {"answer": answer, "cachekey": "metadata"}

    except Exception as e:
        logger.error(f"Metadata chatbot error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Metadata chatbot error: {str(e)}")

@router.post("/chatbot-file", tags=["Contract Chatbot"])
async def chatbot_file(
    file: UploadFile = File(...),
    question: str = Form(...),
    model: str = Form(...)
):
    try:
        # Step 0: Greetings
        if is_greeting(question):
            return {
                "answer": "Hi, I am Contract Genie. I can help you with metadata queries (countries, domains, clauses) or your uploaded contract.",
                "cachekey": "greeting"
            }

        # Step 1: Read file + hash
        file_bytes = await file.read()
        pdf_hash = compute_pdf_hash(file_bytes)

        # Step 2: Cache check
        cache_key = f"{pdf_hash}_{question}_{model}"
        cached_response = db.contracts_cache.find_one({"_id": cache_key})
        if cached_response and cached_response.get("answer"):
            return {"answer": cached_response["answer"], "cachekey": pdf_hash}

        # Step 3: Query type classification
        is_metadata_q = is_metadata_query(question)
        is_contract_q = is_contract_related(question)
        answer = ""
        
        if is_metadata_q:
            intent = detect_intent(question)
            countries, domains, clauses = search_metadata_entities(question)
            answer = format_metadata_response(question, intent, countries, domains, clauses)

        # --- Contract-related queries ---
        elif is_contract_q:
            # Build FAISS index if not cached
            if pdf_hash not in vector_store_cache:
                text = extract_text_from_pdf(file_bytes)
                if text.strip():
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
                    chunks = text_splitter.split_text(text)
                    vector_store = FAISS.from_texts(chunks, embedding_model)
                    vector_store_cache[pdf_hash] = vector_store
            vector_store = vector_store_cache.get(pdf_hash)

            context = ""
            if vector_store:
                retriever = vector_store.as_retriever(search_kwargs={"k": 2})
                docs = retriever.invoke(question)
                context = "\n\n".join([d.page_content for d in docs])

            if context.strip():
                prompt = f"""
                You are an expert contract assistant. Answer the question based ONLY on the provided excerpts.

                Context:
                {context}

                Question:
                {question}

                Instructions:
                - Be concise, plain text.
                - If not found in the context, reply: "Sorry I can't answer this query, try another query."
                """
                try:
                    raw_response = query_endpoint(prompt)
                    # If LLM returns plain text, just use it
                    if isinstance(raw_response, str):
                        answer = raw_response.strip()
                    elif isinstance(raw_response, dict) and "response" in raw_response:
                        answer = raw_response["response"]
                    else:
                        answer = str(raw_response)
                except:
                    answer = "Sorry, I couldn't process the contract content."

        # --- Fallback ---
        else:
            answer = "Sorry, I can't answer this query based on the contract or metadata."

        # Step 4: Cache
        if is_valid_answer(answer):
            db.contracts_cache.update_one(
                {"_id": cache_key},
                {"$set": {"answer": answer, "model": model, "question": question}},
                upsert=True
            )

        return {"answer": answer, "cachekey": pdf_hash}

    except Exception as e:
        logger.error(f"Chatbot error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chatbot error: {str(e)}")

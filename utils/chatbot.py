import re
import json
import re
from typing import List, Dict
from config import db

def normalize_query(q: str) -> str:
    """
    Normalize synonyms in the query so matching works consistently.
    """
    replacements = {
        # Domain synonyms
        r"\bindustry\b": "domain",
        r"\bindustries\b": "domains",
        r"\bsector\b": "domain",
        r"\bsectors\b": "domains",

        # Country synonyms
        r"\bnation\b": "country",
        r"\bnations\b": "countries",
        r"\bstates?\b": "countries",
        r"\bregion\b": "country",
        r"\bregions\b": "countries",

        # Clause synonyms
        r"\bprovision\b": "clause",
        r"\bprovisions\b": "clauses",
        r"\barticle\b": "clause",
        r"\barticles\b": "clauses",
        r"\bterm\b": "clause",
        r"\bterms\b": "clauses",
    }

    for pattern, repl in replacements.items():
        q = re.sub(pattern, repl, q, flags=re.IGNORECASE)

    return q.lower().strip()

def clean_entity_name(name: str) -> str:
    """
    Remove trailing keywords like domain, industry, clause, etc.
    Example: "oil and gas domain" -> "oil and gas"
    """
    stopwords = [
        "domain", "domains", "industry", "industries",
        "sector", "sectors", "country", "countries",
        "clause", "clauses", "provision", "provisions",
        "article", "articles", "term", "terms"
    ]
    tokens = name.split()
    if tokens and tokens[-1].lower() in stopwords:
        tokens = tokens[:-1]
    return " ".join(tokens).strip()

def is_greeting(question: str) -> bool:
    """Check if the question is a greeting"""
    greetings = [
        r'\b(hi|hello|hey|good morning|good afternoon|good evening)\b',
        r'^\s*(hi|hello|hey)\s*[!.?]*\s*$'
    ]
    question_lower = question.lower().strip()
    return any(re.search(pattern, question_lower, re.IGNORECASE) for pattern in greetings)

def is_contract_related(question: str) -> bool:
    q = question.lower().strip()

    # Strong contract indicators
    contract_phrases = [
        "in the contract", "does the contract have", "is there any",
        "under this contract", "mentioned in the contract"
    ]

    # Contract-related keywords (weaker signals)
    contract_keywords = [
        "termination", "payment", "confidentiality", "governing law",
        "liability", "obligation", "condition", "agreement", "sub-clause",
        "section", "rights", "responsibilities"
    ]

    # 1. If any strong phrase is present → definitely contract
    if any(phrase in q for phrase in contract_phrases):
        return True

    # 2. If keywords appear *with contract context words*
    if any(kw in q for kw in contract_keywords):
        if "clause" in q or "contract" in q or "agreement" in q:
            return True

    return False

def is_metadata_query(question: str) -> bool:
    """Check if the question is related to metadata (countries, domains, clauses)"""
    question = question.lower().strip()
    metadata_intents = [
        "list", "show", "give me", "describe", "summary", "summarize",
        "details of", "what are", "explain", "description of", "information on"
    ]
    metadata_keywords = [
        'country', 'countries', 'domain', 'domains', 'clause', 'clauses',
        'what countries', 'which countries', 'list countries', 'show countries',
        'what domains', 'which domains', 'list domains', 'show domains',
        'what clauses', 'which clauses', 'list clauses', 'show clauses',
        'available countries', 'available domains', 'available clauses',
        'force majeure', 'liability', 'oil and gas', 'metadata'
    ]

    if any(intent in question for intent in metadata_intents):
        if any(word in question for word in metadata_keywords):
            return True

    # General metadata lookups
    if "metadata" in question.lower():
        return True

    return False
    
def detect_intent(question: str) -> str:
    """Detect whether the user wants list, count, existence, or detail"""
    q = question.lower()

    if any(kw in q for kw in ["how many", "number of", "count of"]):
        return "count"
    if any(kw in q for kw in ["are there", "is there", "does", "do "]):
        return "existence"
    if any(kw in q for kw in ["list", "show", "which", "available", "what"]):
        return "list"
    if any(kw in q for kw in ["explain", "describe", "summarize", "detail", "tell me about", "give me", "description", "summary", "information on"]):
        return "detail"
    return "unknown"


def search_metadata_entities(question: str) -> tuple:
    """
    Return (countries, domains, clauses) based on entity keywords in question,
    with robust joins (countries → domains → clauses).
    """
    q = normalize_query(question)   # normalize synonyms first

    # --- MULTI-JOIN ---
    # Example: "how many clauses are there for USA in oil and gas"
    if "clause" in q and ("for" in q and "in" in q):
        match = re.search(r'for (.+?) in (.+)', q)
        if match:
            country_name = clean_entity_name(match.group(1).strip())
            domain_name = clean_entity_name(match.group(2).strip())

            pipeline = [
                {"$lookup": {
                    "from": "domains",
                    "localField": "domain_id",
                    "foreignField": "_id",
                    "as": "domain_info"
                }},
                {"$unwind": "$domain_info"},
                {"$lookup": {
                    "from": "countries",
                    "localField": "domain_info.country_id",
                    "foreignField": "_id",
                    "as": "country_info"
                }},
                {"$unwind": "$country_info"},
                {"$match": {
                    "domain_info.domain_name": {"$regex": domain_name, "$options": "i"},
                    "country_info.country_name": {"$regex": country_name, "$options": "i"}
                }}
            ]
            clauses = list(db.clauses.aggregate(pipeline))
            return [], [], clauses

    # --- SINGLE-JOIN ---
    # Countries for a domain
    if "countries" in q and "for" in q:
        domain_match = re.search(r'for (.+)', q)
        if domain_match:
            domain_name = clean_entity_name(domain_match.group(1).strip())
            pipeline = [
                {"$match": {"domain_name": {"$regex": domain_name, "$options": "i"}}},
                {"$lookup": {
                    "from": "countries",
                    "localField": "country_id",
                    "foreignField": "_id",
                    "as": "country_info"
                }},
                {"$unwind": "$country_info"}
            ]
            domains = list(db.domains.aggregate(pipeline))
            countries = [d["country_info"] for d in domains if "country_info" in d]
            return countries, [], []

    # Clauses for a domain
    if "clauses" in q and "for" in q:
        domain_match = re.search(r'for (.+)', q)
        if domain_match:
            domain_name = clean_entity_name(domain_match.group(1).strip())
            pipeline = [
                {"$lookup": {
                    "from": "domains",
                    "localField": "domain_id",
                    "foreignField": "_id",
                    "as": "domain_info"
                }},
                {"$unwind": "$domain_info"},
                {"$match": {"domain_info.domain_name": {"$regex": domain_name, "$options": "i"}}}
            ]
            clauses = list(db.clauses.aggregate(pipeline))
            return [], [], clauses

    # Domains in a country
    if "domains" in q and "in" in q:
        country_match = re.search(r'in (.+)', q)
        if country_match:
            country_name = clean_entity_name(country_match.group(1).strip())
            country = db.countries.find_one({
                "country_name": {"$regex": country_name, "$options": "i"}
            })
            if country:
                domains = list(db.domains.find({"country_id": country["_id"]}))
                return [], domains, []

    # --- DIRECT COLLECTION QUERIES ---
    if "country" in q or "countries" in q:
        return list(db.countries.find({})), [], []
    if "domain" in q or "domains" in q:
        return [], list(db.domains.find({})), []
    if "clause" in q or "clauses" in q:
        return [], [], list(db.clauses.find({}))

    return [], [], []

def format_metadata_response(question: str, intent: str, countries: List[Dict], domains: List[Dict], clauses: List[Dict]) -> str:
    """Format metadata results based on intent classification, with join handling"""

    # --- COUNT intent ---
    if intent == "count":
        print("Count intent detected")
        if "country" in question.lower() or "countries" in question.lower():
            print("Counting countries")
            if countries:
                print("Countries found:", countries)
                return f"There {'is' if len(countries)==1 else 'are'} {len(countries)} country{'s' if len(countries)>1 else ''}: " + ", ".join(c.get("country_name","Unknown") for c in countries)
            return "There are 0 countries."
        if "domain" in question.lower() or "domains" in question.lower():
            print("Counting domains")
            if domains:
                print("Domains found:", domains)
                return f"There {'is' if len(domains)==1 else 'are'} {len(domains)} domain{'s' if len(domains)>1 else ''}: " + ", ".join(d.get("domain_name","Unknown") for d in domains)
            return "There are 0 domains."
        if "clause" in question.lower() or "clauses" in question.lower():
            if clauses:
                return f"There {'is' if len(clauses)==1 else 'are'} {len(clauses)} clause{'s' if len(clauses)>1 else ''}: " + ", ".join(c.get("clause_name","Unknown") for c in clauses)
            return "There are 0 clauses."
        return "No matching metadata found to count."

        # --- EXISTENCE intent ---
    if intent == "existence":
        if countries: return f"There {'is' if len(countries)==1 else 'are'} {len(countries)} matching countr{'y' if len(countries)==1 else 'ies'}."
        if domains: return f"There {'is' if len(domains)==1 else 'are'} {len(domains)} matching domain{'s' if len(domains)!=1 else ''}."
        if clauses: return f"There {'is' if len(clauses)==1 else 'are'} {len(clauses)} matching clause{'s' if len(clauses)!=1 else ''}."
        return "No matching results found."

    # --- DETAIL intent ---
    if intent == "detail" and clauses:
        q = question.lower()
        filtered_clauses = [
            c for c in clauses 
            if any(kw in c.get("clause_name", "").lower() for kw in q.split())
        ]

        # If no exact match, fallback to all
        target_clauses = filtered_clauses if filtered_clauses else clauses

        details = []
        for c in target_clauses:
            details.append({
                "clause_name": c.get("clause_name", "Unknown"),
                # "domain": (c.get("domain_info") or [{}])[0].get("domain_name", "Unknown"),
                # "country": (c.get("country_info") or [{}])[0].get("country_name", "Unknown"),
                "clause_summary": c.get("clause_text", "")[:300] + "..."
            })

        #details = []
        #for c in clauses:
        #    details.append({
        #        "name": c.get("clause_name", "Unknown"),
        #        "domain": (c.get("domain_info") or [{}])[0].get("domain_name", "Unknown"),
        #        "country": (c.get("country_info") or [{}])[0].get("country_name", "Unknown"),
        #        "text": c.get("clause_text", "")[:300] + "..."
        #    })
        return json.dumps(details, indent=2)

    # --- LIST intent (fallback if no details) ---
    if intent in ["list", "unknown"]:
        if countries: return "Countries: " + ", ".join(c.get("country_name","Unknown") for c in countries)
        if domains: return "Domains: " + ", ".join(d.get("domain_name","Unknown") for d in domains)
        if clauses: return "Clauses: " + ", ".join(c.get("clause_name","Unknown") for c in clauses)

    return "No metadata results found."

def is_valid_answer(answer: str) -> bool:
    if not answer or not answer.strip():
        return False

    # Whitelisted safe fallback responses
    safe_fallbacks = [
        "sorry, i can't answer this query based on the contract or metadata"
    ]
    if answer.strip().lower() in safe_fallbacks:
        return True

    # General error-like patterns
    error_patterns = [
        r'(^|\n)\s*error',
        r'\bsorry[, ]',             
        r'couldn\'t process',       
        r'can[\' ]?t answer',       
        r'not able to',             
        r'failed to',               
        r'exception',               
        r'invalid',                 
        r'json.*error',             
    ]

    ans_lower = answer.lower()
    for pattern in error_patterns:
        if re.search(pattern, ans_lower, flags=re.IGNORECASE):
            return False

    return True

from fastapi import APIRouter, HTTPException, Query

#Import Models
from models import Country, Domain, Compliance, Clause

# Import Config
from config import db

router = APIRouter()

def get_next_sequence(name):
    counter = db.counters.find_one_and_update(
        {"_id": name},
        {"$inc": {"sequence_value": 1}},
        upsert=True,
        return_document=True
    )
    return counter["sequence_value"]

# ---------- CRUD for Country ----------
@router.post("/countries",tags=["Countries"])
def create_country(country: Country):
    next_id = get_next_sequence("country_id")
    db.countries.insert_one({"_id": next_id, "country_name": country.country_name})
    return {"message": "Country added", "id": next_id}

@router.get("/countries",tags=["Countries"])
def get_countries():
    return list(db.countries.find({}, {"_id": 1, "country_name": 1}))

@router.put("/countries/{country_id}",tags=["Countries"])
def update_country(country_id: int, country: Country):
    result = db.countries.update_one({"_id": country_id}, {"$set": {"country_name": country.country_name}})
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Country not found")
    return {"message": "Country updated"}

@router.delete("/countries/{country_id}",tags=["Countries"])
def delete_country(country_id: int):
    result = db.countries.delete_one({"_id": country_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Country not found")
    return {"message": "Country deleted"}

@router.get("/countries/{country_id}", tags=["Countries"])
def get_country_by_id(country_id: int):
    country = db.countries.find_one({"_id": country_id}, {"_id": 1, "country_name": 1})
    if not country:
        raise HTTPException(status_code=404, detail="Country not found")
    return country

#----- CRUD for Domain ----------
@router.post("/domains",tags=["Domains"])
def create_domain(domain: Domain):
    next_id = get_next_sequence("domain_id")
    db.domains.insert_one({
        "_id": next_id,
        "domain_name": domain.domain_name,
        "country_id": domain.country_id
    })
    return {"message": "Domain added", "id": next_id}

@router.get("/domains", tags=["Domains"])
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

@router.get("/domains/byid/{domain_id}", tags=["Domains"])
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

@router.put("/domains/{domain_id}",tags=["Domains"])
def update_domain(domain_id: int, domain: Domain):
    result = db.domains.update_one({"_id": domain_id}, {"$set": {
        "domain_name": domain.domain_name,
        "country_id": domain.country_id
    }})
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Domain not found")
    return {"message": "Domain updated"}

@router.delete("/domains/{domain_id}",tags=["Domains"])
def delete_domain(domain_id: int):
    result = db.domains.delete_one({"_id": domain_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Domain not found")
    return {"message": "Domain deleted"}

@router.get("/domains/bycountry/{country_id}", tags=["Domains"])
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

# ----- CRUD for Compliance ----------
@router.get("/compliances", tags=["Compliances"])
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

@router.post("/compliances",tags=["Compliances"])
def create_compliance(compliance: Compliance):
    next_id = get_next_sequence("compliance_id")
    db.compliances.insert_one({
        "_id": next_id,
        "compliance_name": compliance.compliance_name,
        "domain_id": compliance.domain_id
    })
    return {"message": "Compliance added", "id": next_id}

@router.get("/compliances/{domain_id}", tags=["Compliances"])
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

@router.put("/compliances/{compliance_id}",tags=["Compliances"])
def update_compliance(compliance_id: int, compliance: Compliance):
    result = db.compliances.update_one({"_id": compliance_id}, {"$set": {
        "compliance_name": compliance.compliance_name,
        "domain_id": compliance.domain_id
    }})
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Compliance not found")
    return {"message": "Compliance updated"}

@router.delete("/compliances/{compliance_id}",tags=["Compliances"])
def delete_compliance(compliance_id: int):
    result = db.compliances.delete_one({"_id": compliance_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Compliance not found")
    return {"message": "Compliance deleted"}

#-------- CRUD for Clause ----------

@router.post("/clauses", tags=["Clauses"])
def create_clause(clause: Clause):
    next_id = get_next_sequence("clause_id")  # Your ID generator
    db.clauses.insert_one({
        "_id": next_id,
        "clause_name": clause.clause_name,
        "clause_text": clause.clause_text,
        "domain_id": clause.domain_id
    })
    return {"message": "Clause added", "id": next_id}

@router.get("/clauses", tags=["Clauses"])
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
            "$lookup": {
                "from": "countries",
                "localField": "domain_info.country_id",
                "foreignField": "_id",
                "as": "country_info"
            }
        },
        {"$unwind": {"path": "$country_info", "preserveNullAndEmptyArrays": True}},
        {
            "$project": {
                "_id": 1,
                "clause_name": 1,
                "clause_text": 1,
                "domain_id": 1,
                "domain_name": "$domain_info.domain_name",
                "country_id": "$country_info._id",
                "country_name": "$country_info.country_name"
            }
        }
    ]

    clauses = list(db.clauses.aggregate(pipeline))
    return clauses

@router.get("/clauses/filter", tags=["Clauses"])
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

@router.get("/clauses/by-domain/{domain_id}", tags=["Clauses"])
def get_clauses_by_domain(domain_id: int):
    clauses = list(db.clauses.find(
        {"domain_id": domain_id},
        {"_id": 1, "clause_name": 1, "clause_text": 1, "domain_id": 1}
    ))
    return clauses

@router.put("/clauses/{clause_id}", tags=["Clauses"])
def update_clause(clause_id: int, clause: Clause):
    result = db.clauses.update_one({"_id": clause_id}, {"$set": {
        "clause_name": clause.clause_name,
        "clause_text": clause.clause_text,
        "domain_id": clause.domain_id
    }})
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Clause not found")
    return {"message": "Clause updated"}

@router.get("/clauses/{clause_id}", tags=["Clauses"])
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

@router.delete("/clauses/{clause_id}", tags=["Clauses"])
def delete_clause(clause_id: int):
    result = db.clauses.delete_one({"_id": clause_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Clause not found")
    return {"message": "Clause deleted"}

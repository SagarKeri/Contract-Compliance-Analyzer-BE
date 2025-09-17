from pydantic import BaseModel, EmailStr, Field
from typing import Optional

#!----- Feedback Input Models -------
class FeedbackInput(BaseModel):
    feedback: str
    cache_key: str

#!----- Metadata Models -------
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

#!----- ChatBot Input Models -------
class ChatbotFileInput(BaseModel):
    question: str
    model: str  # "1" = Mistral, "2" = Gemini

class ChatbotMetadataInput(BaseModel):
    question: str
    model: str

#!----- Authentication Models -------
class UserCreate(BaseModel):
    first_name: str
    last_name: str
    email: EmailStr
    password: str
    role_id: int = 2  # Default to 2 (User)

class UserInDB(BaseModel):
    first_name: str
    last_name: str
    email: EmailStr
    hashed_password: str
    role_id: int
    role_name: str

class UserInfo(BaseModel):
    id: str = Field(..., alias="_id")
    first_name: str
    last_name: str
    email: str
    role_id: int
    role_name: str

    class Config:
        allow_population_by_field_name = True

class Token(BaseModel):
    access_token: str
    token_type: str
    user: UserInfo

class TokenData(BaseModel):
    email: Optional[str] = None
    role: Optional[str] = None

class UpdateUserRole(BaseModel):
    user_id: str
    role_id: int
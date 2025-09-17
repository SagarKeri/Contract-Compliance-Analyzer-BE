from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pymongo.errors import DuplicateKeyError
from bson import ObjectId

#Import Models
from models import UserCreate, Token, TokenData, UpdateUserRole

#Import Helper Functions
from utils.authentication import get_role, hash_password, verify_password, create_access_token, get_user, get_current_user, get_current_admin

#Import Config
from config import db

router = APIRouter()

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")

@router.post("/signup", tags=["Authentication"])
async def signup(user: UserCreate):
    """Register a new user with role_id (1 for Admin, 2 for User, defaults to 2)."""
    role = get_role(user.role_id)
    if not role:
        raise HTTPException(status_code=400, detail="Invalid role_id. Must be 1 (Admin) or 2 (User).")
    if get_user(user.email):
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed_password = hash_password(user.password)
    user_dict = {
        "first_name": user.first_name,
        "last_name": user.last_name,
        "email": user.email,
        "hashed_password": hashed_password,
        "role_id": user.role_id,
        "role_name": role["role_name"]
    }
    try:
        db.users.insert_one(user_dict)
    except DuplicateKeyError:
        raise HTTPException(status_code=400, detail="Email already registered")
    return {"message": "User registered successfully"}

@router.post("/login", response_model=Token, tags=["Authentication"])
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Authenticate user and return JWT token."""
    user = get_user(form_data.username)  # form_data.username is the email
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(
        data={"sub": user["email"], "role": user["role_name"]}
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "_id": str(user["_id"]),
            "first_name": user["first_name"],
            "last_name": user["last_name"],
            "email": user["email"],
            "role_id": user["role_id"],
            "role_name": user["role_name"],
        }
    }

@router.get("/users", tags=["Authentication"])
async def get_all_users(current_admin: dict = Depends(get_current_admin)):
    """
    Get all registered users.
    Only accessible by Admin.
    """
    users = list(db.users.find({}, {"hashed_password": 0}))  

    # Convert ObjectId to string
    for user in users:
        user["_id"] = str(user["_id"])

    return {"users": users}

@router.delete("/users/{user_id}", tags=["Authentication"])
async def delete_user(user_id: str, current_admin: dict = Depends(get_current_admin)):
    """
    Delete a user by ID.
    Only accessible by Admin.
    """
    # Validate ObjectId
    if not ObjectId.is_valid(user_id):
        raise HTTPException(status_code=400, detail="Invalid user ID format")

    result = db.users.delete_one({"_id": ObjectId(user_id)})

    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="User not found")

    return {"message": "User deleted successfully", "user_id": user_id}


@router.put("/users/update-role", tags=["Authentication"])
async def update_user_role(
    data: UpdateUserRole,
    current_admin: dict = Depends(get_current_admin)
    ):
    """
    Update a user's role (Admin-only).
    Requires user_id and new role_id.
    """
    # Validate role
    role = get_role(data.role_id)
    if not role:
        raise HTTPException(status_code=400, detail="Invalid role_id. Must be 1 (Admin) or 2 (User).")

    # Convert string user_id to ObjectId
    try:
        user_obj_id = ObjectId(data.user_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid user_id format")

    # Find user
    user = db.users.find_one({"_id": user_obj_id})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Update role
    db.users.update_one(
        {"_id": user_obj_id},
        {"$set": {"role_id": data.role_id, "role_name": role["role_name"]}}
    )

    return {"message": f"User role updated to {role['role_name']} successfully"}



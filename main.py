from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import routers from routes package
from routes import *

# ---------- Initialize FastAPI App ---------- #
app = FastAPI(
    title="Oil & Gas Contract Compliance API",
    description="Upload oil & gas contracts in PDF format and get compliance analysis across 19 regulatory areas.",
    version="1.6.0",
)

# ---------- CORS Middleware ---------- #
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)

# ---------- Include Routers ---------- #
app.include_router(analyze.router)
app.include_router(metadata.router)
app.include_router(chatbot.router)
app.include_router(authentication.router)
app.include_router(contracts_cache.router)
app.include_router(user_analysis.router)

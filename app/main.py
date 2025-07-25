from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.routes import router as core_router
from app.routes_accidents import router as accidents_router
from app.routes_speed import router as speed_router
from app.routes_fraud import router as fraud_router
import os

app = FastAPI(title="Traffic Backend")

os.makedirs("outputs", exist_ok=True)

app.mount("/static", StaticFiles(directory="outputs"), name="static")


# Allow your frontend origin
origins = [
    "http://localhost:5173",  # Vite/React dev server
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

app.include_router(core_router)
app.include_router(accidents_router)
app.include_router(speed_router)
app.include_router(fraud_router)

@app.get("/")
def root():
    return {"message": "Traffic API is running!"}
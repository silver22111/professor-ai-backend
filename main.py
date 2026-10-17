import os
import httpx
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext

app = FastAPI(title="🎓 Professor AI Secure Backend Engine")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
user_database = {}

class UserRegister(BaseModel):
    name: str
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class ChatPayload(BaseModel):
    message: str

@app.post("/api/auth/signup", status_code=status.HTTP_201_CREATED)
async def signup(user: UserRegister):
    if user.email in user_database:
        raise HTTPException(status_code=400, detail="Account already registered under this email string.")
    hashed_password = pwd_context.hash(user.password)
    user_database[user.email] = {"name": user.name, "password": hashed_password}
    return {"message": "Account registered securely. Proceed to login interface terminal."}

@app.post("/api/auth/signin")
async def signin(user: UserLogin):
    db_user = user_database.get(user.email)
    if not db_user or not pwd_context.verify(user.password, db_user["password"]):
        raise HTTPException(status_code=400, detail="Invalid credential parameters passed.")
    return {"message": "Authentication successful.", "user": {"name": db_user["name"], "email": user.email}}

@app.post("/api/ai/chat")
async def process_chat(payload: ChatPayload):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Server Gemini environment keys missing configuration signatures.")
    gemini_endpoint = f"https://googleapis.com{GEMINI_API_KEY}"
    formatted_payload = {"contents": [{"parts": [{"text": payload.message}]}]}
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(gemini_endpoint, json=formatted_payload, timeout=10.0)
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail="AI Brain terminal handshake rejected.")
            data = response.json()
            ai_text = data["candidates"][0]["content"]["parts"][0]["text"]
            return {"response": ai_text.strip()}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Internal proxy execution breakdown: {str(e)}")

import pandas as pd
from fastapi import FastAPI, Request, Form, Depends, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
from starlette.middleware.sessions import SessionMiddleware
from sqlalchemy import create_engine, Column, String, Boolean, Integer, Float, inspect, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from models.NABIL_predict import main as predict_main_nabil
from models.ADBL_predict import main as predict_main_adbl
from models.NMB_predict import main as predict_main_nmb
import bcrypt

# Database setup
DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# User model
class User(Base):
    __tablename__ = "users"
    username = Column(String, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    full_name = Column(String)
    hashed_password = Column(String)
    disabled = Column(Boolean, default=False)

# Create tables (only users table is created here)
Base.metadata.create_all(bind=engine)

app = FastAPI()

# Add SessionMiddleware with a secret key
app.add_middleware(SessionMiddleware, secret_key="your-secret-key")

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

# Pydantic models
class UserSignup(BaseModel):
    username: str
    email: str
    full_name: str
    password: str

class UserResponse(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None

# Function to hash password
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

# Function to verify password
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Function to get the list of tables
def get_tables():
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    return [table for table in tables if table != 'users']

# Homepage
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request, model: str = 'GCN', bank: str = 'NABIL'):
    try:
        if bank == 'NABIL':
            graphs = predict_main_nabil(model)  
        elif bank == 'ADBL':
            graphs = predict_main_adbl(model) 
        elif bank == 'NMB':
            graphs = predict_main_nmb(model) 
        else:
            raise HTTPException(status_code=400, detail='Invalid bank specified')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating predictions: {str(e)}")
    
    return templates.TemplateResponse("home.html", {"request": request, "graphs": graphs, "model": model, "bank": bank})

# NEPSE Data page
@app.get("/nepse-data", response_class=HTMLResponse)
async def nepse_data(request: Request):
    tables = get_tables()
    categories = {
        "Commercial Bank": ["ADBL", "CZBIL", "EBL", "GBIME", "HBL", "KBL", "MBL", "NABIL", "NBL", "SBI", "NICA", "NMB", "PRVU", "PCBL", "SANIMA", "SBL", "SCB"],
        "Development Bank": ["CORBL", "EDBL", "GBBL", "GRDBL", "JBBL", "KSBBL", "LBBL", "MLBL", "MDB", "MNBBL", "NABBC", "SAPDBL", "SADBL", "SHINE", "SINDU"],
        "Finance": ["BFC", "CFCL", "GFCL", "ICFC", "JFL", "MFIL", "MPFL", "NFS", "PFL", "PROFL", "RLFL", "SFCL", "SIFC"]
    }
    return templates.TemplateResponse("nepse_data.html", {"request": request, "tables": tables, "categories": categories})

# View table data
@app.get("/tables/{table_name}", response_class=HTMLResponse)
async def view_table(request: Request, table_name: str, db: SessionLocal = Depends(get_db)):
    # Fetch data from the table using a raw SQL query
    query = text(f"SELECT * FROM {table_name}")  
    result = db.execute(query).fetchall()

    data = [row._asdict() for row in result]

    # Pass the data to the template
    return templates.TemplateResponse("view_table.html", {"request": request, "table_name": table_name, "data": data})

# Login page
@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

# Login functionality
@app.post("/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...), db: SessionLocal = Depends(get_db)):
    user = db.query(User).filter(User.username == username).first()
    if not user or not verify_password(password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    
    # Set the username in the session
    request.session["username"] = username
    return RedirectResponse(url="/", status_code=303)

# Signup page
@app.get("/signup", response_class=HTMLResponse)
async def signup_page(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})

# Signup functionality
@app.post("/signup")
async def signup(request: Request, username: str = Form(...), email: str = Form(...), full_name: str = Form(...), password: str = Form(...), db: SessionLocal = Depends(get_db)):
    if db.query(User).filter(User.username == username).first():
        raise HTTPException(status_code=400, detail="Username already exists")
    
    # Hash the password and create a new user
    hashed_password = hash_password(password)
    new_user = User(username=username, email=email, full_name=full_name, hashed_password=hashed_password)
    db.add(new_user)
    db.commit()
    
    # Set the username in the session
    request.session["username"] = username
    return RedirectResponse(url="/", status_code=303)

# Logout functionality
@app.get("/logout")
async def logout(request: Request):
    # Clear the session
    request.session.pop("username", None)
    # Redirect to the homepage
    return RedirectResponse(url="/", status_code=303)

    
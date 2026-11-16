import joblib
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from datetime import datetime, timedelta, timezone
from jose import jwt
from database import Base, engine, get_db
from models import User
from schemas import UserRegister, UserLogin, UserResponse
from device_models import Device
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from device_schemas import DeviceRegister, DeviceResponse
from flow_models import NetworkFlow
from flow_schemas import FlowBatch
from pydantic import BaseModel

# Load trained Zeek anomaly detection pipeline
PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_DIR / "zeek_anomaly_model.pkl"

anomaly_artifact = joblib.load(MODEL_PATH)

anomaly_pipeline = anomaly_artifact["pipeline"]

ANOMALY_FEATURES = anomaly_artifact["features"]

print("Zeek AI anomaly model loaded successfully!")
print("Expected features:", ANOMALY_FEATURES)

app = FastAPI(
    title="AI Network Security API",
    description="Centralized Network Intrusion Detection System",
    version="1.0.0"
)

# Create database tables
Base.metadata.create_all(bind=engine)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT configuration
SECRET_KEY = "CHANGE_THIS_TO_A_LONG_RANDOM_SECRET_BEFORE_DEPLOYMENT"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

security = HTTPBearer()


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    token = credentials.credentials

    try:
        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM]
        )
        user_id = int(payload.get("sub"))

    except Exception:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token"
        )

    user = db.query(User).filter(User.id == user_id).first()

    if not user:
        raise HTTPException(
            status_code=401,
            detail="User not found"
        )

    return user



def create_access_token(user_id: int):
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=ACCESS_TOKEN_EXPIRE_MINUTES
    )

    payload = {
        "sub": str(user_id),
        "exp": expire
    }

    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


@app.get("/")
def home():
    return {
        "message": "AI Network Security Backend is running!",
        "status": "success"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "FastAPI Backend"
    }


@app.post("/register", response_model=UserResponse)
def register_user(data: UserRegister, db: Session = Depends(get_db)):

    existing_user = db.query(User).filter(
        (User.email == data.email) |
        (User.username == data.username)
    ).first()

    if existing_user:
        raise HTTPException(
            status_code=409,
            detail="Email or username already registered"
        )

    hashed_password = pwd_context.hash(data.password)

    new_user = User(
        username=data.username,
        email=data.email,
        hashed_password=hashed_password
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return new_user


@app.post("/login")
def login_user(data: UserLogin, db: Session = Depends(get_db)):

    user = db.query(User).filter(
        User.email == data.email
    ).first()

    if not user or not pwd_context.verify(
        data.password, user.hashed_password
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password"
        )

    token = create_access_token(user.id)

    return {
        "message": "Login successful",
        "access_token": token,
        "token_type": "bearer",
        "user_id": user.id,
        "username": user.username
    }

@app.post("/devices/register", response_model=DeviceResponse)
def register_device(
    data: DeviceRegister,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    existing_device = db.query(Device).filter(
        Device.user_id == current_user.id,
        Device.hostname == data.hostname
    ).first()

    if existing_device:
        raise HTTPException(
            status_code=409,
            detail="This device is already registered"
        )

    new_device = Device(
        user_id=current_user.id,
        device_name=data.device_name,
        hostname=data.hostname,
        operating_system=data.operating_system
    )

    db.add(new_device)
    db.commit()
    db.refresh(new_device)

    return new_device


@app.get("/devices", response_model=list[DeviceResponse])
def get_my_devices(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    return db.query(Device).filter(
        Device.user_id == current_user.id
    ).all()

@app.post("/sensor/ingest")
def ingest_flows(
    batch: FlowBatch,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    device = db.query(Device).filter(
        Device.id == batch.device_id,
        Device.user_id == current_user.id,
        Device.is_active.is_(True)
    ).first()

    if not device:
        raise HTTPException(
            status_code=404,
            detail="Registered active device not found"
        )

    for flow in batch.flows:
        flow_data = flow.model_dump()

        # Prepare the exact 5 features expected by the trained model
        feature_data = {
            feature: flow_data.get(feature)
            for feature in ANOMALY_FEATURES
        }

        feature_df = pd.DataFrame(
            [feature_data],
            columns=ANOMALY_FEATURES
        )

        # Run Isolation Forest prediction
        prediction_value = int(
            anomaly_pipeline.predict(feature_df)[0]
        )

        # Higher score means more unusual
        anomaly_score = float(
            -anomaly_pipeline.decision_function(feature_df)[0]
        )

        prediction_label = (
            "UNUSUAL" if prediction_value == -1 else "BASELINE"
        )

        record = NetworkFlow(
            device_id=device.id,
            **flow_data,
            anomaly_score=anomaly_score,
            prediction=prediction_label
        )

        db.add(record)

    db.commit()

    return {
        "message": "Flow records received successfully",
        "device_id": device.id,
        "records_saved": len(batch.flows)
    }
@app.get("/my-flows")
def get_my_flows(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    flows = (
        db.query(NetworkFlow)
        .join(Device, NetworkFlow.device_id == Device.id)
        .filter(Device.user_id == current_user.id)
        .order_by(NetworkFlow.id.desc())
        .limit(500)
        .all()
    )

    return [
        {
            "id": flow.id,
            "device_id": flow.device_id,
            "source_ip": flow.source_ip,
            "destination_ip": flow.destination_ip,
            "protocol": flow.protocol,
            "duration": flow.duration,
            "orig_bytes": flow.orig_bytes,
            "resp_bytes": flow.resp_bytes,
            "prediction": flow.prediction,
            "anomaly_score": flow.anomaly_score,
            "created_at": str(flow.created_at)
        }
        for flow in flows
    ]
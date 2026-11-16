
from pydantic import BaseModel, Field


class DeviceRegister(BaseModel):
    device_name: str = Field(min_length=2, max_length=100)
    hostname: str = Field(min_length=1, max_length=150)
    operating_system: str = Field(min_length=2, max_length=100)


class DeviceResponse(BaseModel):
    id: int
    user_id: int
    device_name: str
    hostname: str
    operating_system: str
    is_active: bool

    class Config:
        from_attributes = True

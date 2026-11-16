
from pydantic import BaseModel, Field
from typing import Optional


class ZeekFlowIn(BaseModel):
    uid: Optional[str] = None
    timestamp: float = Field(gt=0)

    source_ip: Optional[str] = None
    source_port: Optional[int] = Field(default=None, ge=0, le=65535)

    destination_ip: Optional[str] = None
    destination_port: Optional[int] = Field(default=None, ge=0, le=65535)

    protocol: Optional[str] = None
    service: Optional[str] = None
    connection_state: Optional[str] = None

    duration: Optional[float] = None

    orig_bytes: Optional[int] = Field(default=None, ge=0)
    resp_bytes: Optional[int] = Field(default=None, ge=0)
    orig_pkts: Optional[int] = Field(default=None, ge=0)
    resp_pkts: Optional[int] = Field(default=None, ge=0)


class FlowBatch(BaseModel):
    device_id: int
    flows: list[ZeekFlowIn]

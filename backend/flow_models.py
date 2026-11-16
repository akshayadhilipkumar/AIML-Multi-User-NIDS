
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    ForeignKey
)
from sqlalchemy.sql import func

from database import Base


class NetworkFlow(Base):
    __tablename__ = "network_flows"

    id = Column(Integer, primary_key=True, index=True)

    # Which registered device generated this flow?
    device_id = Column(
        Integer,
        ForeignKey("devices.id"),
        nullable=False,
        index=True
    )

    # Zeek connection details
    uid = Column(String(100), nullable=True)
    timestamp = Column(Float, nullable=False)

    source_ip = Column(String(45), nullable=True)
    source_port = Column(Integer, nullable=True)

    destination_ip = Column(String(45), nullable=True)
    destination_port = Column(Integer, nullable=True)

    protocol = Column(String(20), nullable=True)
    service = Column(String(50), nullable=True)
    connection_state = Column(String(20), nullable=True)

    duration = Column(Float, nullable=True)
    orig_bytes = Column(Integer, nullable=True)
    resp_bytes = Column(Integer, nullable=True)

    orig_pkts = Column(Integer, nullable=True)
    resp_pkts = Column(Integer, nullable=True)

    # AI analysis results
    anomaly_score = Column(Float, nullable=True)
    prediction = Column(String(50), nullable=True)

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now()
    )

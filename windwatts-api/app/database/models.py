from sqlalchemy import Column, Integer, String, DateTime, JSON, Text
from sqlalchemy.sql import func
from .connection import Base


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String, nullable=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    user_id = Column(String, nullable=True, index=True)
    action = Column(String, index=True)
    resource = Column(String, index=True)
    method = Column(String, index=True)
    status_code = Column(Integer, nullable=True, index=True)
    ip_address = Column(String, nullable=True, index=True)
    user_agent = Column(Text, nullable=True)
    duration_ms = Column(Integer, nullable=True, index=True)
    request_size_bytes = Column(Integer, nullable=True)
    response_size_bytes = Column(Integer, nullable=True)
    log_metadata = Column(JSON, nullable=True)

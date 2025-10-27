"""
Schemas Pydantic pour les licences
"""

from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List


class LicenseResponse(BaseModel):
    """Schema pour la réponse licence"""
    id: str
    license_key: str
    license_type: str
    status: str
    features: Optional[List[str]]
    quota_daily_analyses: int
    quota_daily_api_calls: int
    activated_at: Optional[datetime]
    expires_at: Optional[datetime]
    days_remaining: Optional[int] = None
    
    class Config:
        from_attributes = True


class LicenseActivate(BaseModel):
    """Schema pour activer une licence"""
    license_key: str
    machine_id: str

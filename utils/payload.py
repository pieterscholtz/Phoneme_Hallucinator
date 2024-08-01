from pydantic import BaseModel
from typing import Optional

class Payload(BaseModel):
    env: str
    input_path: str
    num_samples: int
    noop_reason: Optional[str] = None

class Task(BaseModel):
    env: str
    input_path: str
    num_samples: int
    output_path: str
    noop_reason: Optional[str] = None

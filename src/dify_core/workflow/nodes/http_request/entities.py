from typing import Literal, Optional
from pydantic import BaseModel

from dify_core.workflow.nodes.base.entities import BaseNodeData


class HttpRequestNodeAuthorization(BaseModel):
    """HTTP request authorization"""
    type: str
    config: dict = {}


class HttpRequestNodeBody(BaseModel):
    """HTTP request body"""
    type: str
    data: str


class HttpRequestNodeTimeout(BaseModel):
    """HTTP request timeout"""
    max_connect_timeout: int = 10
    max_read_timeout: int = 60
    max_write_timeout: int = 20


class HttpRequestNodeData(BaseNodeData):
    """HTTP Request Node Data."""
    method: Literal["get", "post", "put", "patch", "delete", "head", "options"]
    url: str
    authorization: HttpRequestNodeAuthorization
    headers: str
    params: str
    body: Optional[HttpRequestNodeBody] = None
    timeout: Optional[HttpRequestNodeTimeout] = None
    ssl_verify: Optional[bool] = True
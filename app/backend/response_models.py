from pydantic import BaseModel


class LocationResponse(BaseModel):
    ip: str
    country: str
    region: str
    city: str
    lat: float
    lon: float


class ImageUploadResponse(BaseModel):
    status: str
    message: str


class ObjectResponse(BaseModel):
    key: str
    size: int
    last_modified: str


class DebugListObjectsResponse(BaseModel):
    objects: list[ObjectResponse]
    count: int

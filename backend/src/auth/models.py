from pydantic import BaseModel, EmailStr


class UserCreate(BaseModel):
    name: str
    company_name: str
    email: EmailStr
    password: str


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    id: str
    name: str
    company_name: str
    email: str


class ChangePassword(BaseModel):
    old_password: str
    new_password: str

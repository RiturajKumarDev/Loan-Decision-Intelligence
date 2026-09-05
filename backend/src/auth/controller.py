from fastapi import APIRouter, HTTPException, Depends, Response
from bson import ObjectId

from src.auth.service import (
    hash_password,
    verify_password,
    create_access_token,
    verify_token,
)
from src.database.core import users_collection
from src.auth.models import (
    UserCreate,
    UserResponse,
    UserLogin,
    ChangePassword,
)

router = APIRouter(
    prefix="/auth",
    tags=["Authentication"],
)


@router.post("/register")
async def register_user(user: UserCreate):
    existing_user = await users_collection.find_one({"email": user.email})
    if existing_user:
        raise HTTPException(
            status_code=400,
            detail="User with this email already exists.",
        )
    hashed_password = hash_password(user.password)
    user_data = {
        "name": user.name,
        "company_name": user.company_name,
        "email": user.email,
        "password": hashed_password,
    }
    result = await users_collection.insert_one(user_data)
    return {
        "success": True,
        "user_id": str(result.inserted_id),
    }


@router.post("/login")
async def login(user: UserLogin):
    current_user = await users_collection.find_one({"email": user.email})
    if not current_user:
        raise HTTPException(
            status_code=404,
            detail="User not found",
        )
    if not verify_password(
        user.password,
        current_user["password"],
    ):
        raise HTTPException(
            status_code=401,
            detail="Incorrect password!!",
        )
    token = create_access_token(
        {
            "sub": current_user["email"],
            "user_id": str(current_user["_id"]),
            "name": current_user["name"],
        }
    )
    return {
        "success": True,
        "access_token": token,
        "token_type": "bearer",
    }


@router.get(
    "/profile",
    response_model=UserResponse,
)
async def get_profile(
    payload: dict = Depends(verify_token),
):
    user = await users_collection.find_one({"email": payload["sub"]})
    if not user:
        raise HTTPException(
            status_code=404,
            detail="User not found",
        )
    return {
        "id": str(user["_id"]),
        "name": user["name"],
        "company_name": user["company_name"],
        "email": user["email"],
    }


@router.get("/logout")
def logout_user(response: Response):
    response.delete_cookie("access_token")
    return {
        "success": True,
        "message": "Logout successful",
    }


@router.put(
    "/change_password",
)
async def change_password(
    request: ChangePassword,
    payload: dict = Depends(verify_token),
):
    user = await users_collection.find_one({"email": payload["sub"]})
    if not user:
        raise HTTPException(
            status_code=404,
            detail="User does not exist!!",
        )
    if not verify_password(
        request.old_password,
        user["password"],
    ):
        raise HTTPException(
            status_code=400,
            detail="Incorrect old password!!",
        )
    new_password = hash_password(request.new_password)
    result = await users_collection.update_one(
        {"_id": user["_id"]},
        {"$set": {"password": new_password}},
    )
    if result.matched_count == 0:
        raise HTTPException(
            status_code=404,
            detail="User not found",
        )
    return {
        "success": True,
        "message": "Password updated successfully",
    }

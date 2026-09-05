from motor.motor_asyncio import AsyncIOMotorClient

from src.core.config import settings

client = AsyncIOMotorClient(settings.DATABASE_URL)

database = client[settings.DATABASE_NAME]


users_collection = database["users"]
histories_collection = database["histories"]

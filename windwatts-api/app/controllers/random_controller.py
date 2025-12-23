"""
This still just exists as an example...we should remove this before we go live
"""

import requests
from fastapi import APIRouter, HTTPException, Query
from app.utils.random_message import random_message as get_random_message

router = APIRouter()

categories = [
    "animal",
    "career",
    "celebrity",
    "dev",
    "explicit",
    "fashion",
    "food",
    "history",
    "money",
    "movie",
    "music",
    "political",
    "religion",
    "science",
    "sport",
    "travel",
]


@router.get("/", summary="Retrieve a random message")
def read_random_message():
    try:
        message = get_random_message()
        return {"message": message}
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error.")


@router.get("/chuck", summary="Get a random Chuck Norris joke")
def get_chuck_norris_joke():
    try:
        response = requests.get("https://api.chucknorris.io/jokes/random", timeout=5)
        response.raise_for_status()
        joke = response.json().get("value")
        return {"joke": joke}
    except requests.RequestException:
        raise HTTPException(
            status_code=502, detail="Failed to fetch joke from external service."
        )
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error.")


@router.get(
    "/chuck/{category}",
    summary="Get a random Chuck Norris joke from a list of categories",
)
def get_chuck_norris_joke_by_category(
    category: str = Query(..., description="Joke category"),
):
    if category not in categories:
        raise HTTPException(
            status_code=400, detail=f"Invalid category. Must be one of: {categories}"
        )
    try:
        response = requests.get(
            f"https://api.chucknorris.io/jokes/random?category={category}", timeout=5
        )
        response.raise_for_status()
        joke = response.json().get("value")
        return {"joke": joke}
    except requests.RequestException:
        raise HTTPException(
            status_code=502, detail="Failed to fetch joke from external service."
        )
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error.")

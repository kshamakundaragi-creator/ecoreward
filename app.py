
import os
import json
import base64
from pathlib import Path
from datetime import datetime, timezone
from threading import Lock

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.responses import FileResponse
from pathlib import Path
from pydantic import BaseModel

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data.json"
LOCK = Lock()

app = FastAPI(
    title="EcoReward LangChain Waste Management",
    description="AI-powered waste classification, rewards and eco-quests",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_data():
    if not DATA_FILE.exists():
        return {
            "users": {
                "USER001": {
                    "name": "User 001",
                    "points": 0,
                    "level": "Eco Beginner"
                },
                "USER002": {
                    "name": "User 002",
                    "points": 0,
                    "level": "Eco Beginner"
                }
            },
            "history": [],
            "pending_submissions": [],
            "quests": [
                {
                    "id": 1,
                    "title": "Recycle a Plastic Bottle",
                    "description": "Submit a plastic bottle for recycling.",
                    "points": 5
                },
                {
                    "id": 2,
                    "title": "Recycle E-Waste",
                    "description": "Submit an old electronic device.",
                    "points": 10
                },
                {
                    "id": 3,
                    "title": "Build a Greener Habit",
                    "description": "Separate recyclable waste at home.",
                    "points": 5
                }
            ]
        }

    with open(DATA_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)
    data.setdefault("pending_submissions", [])
    return data


def save_data(data):
    with open(DATA_FILE, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def get_level(points):
    if points >= 100:
        return "Eco Champion"
    if points >= 50:
        return "Green Guardian"
    if points >= 20:
        return "Eco Explorer"
    return "Eco Beginner"


def get_profile(user_id):
    data = load_data()
    user = data["users"].get(user_id)
    if not user:
        return {"error": f"User {user_id} was not found."}

    return {"user_id": user_id, **user}


def update_points(user_id, points):
    data = load_data()
    user = data["users"].get(user_id)

    if not user:
        return {"error": f"User {user_id} was not found."}

    user["points"] += points
    user["level"] = get_level(user["points"])
    save_data(data)

    return {
        "user_id": user_id,
        "points_added": points,
        "total_points": user["points"],
        "level": user["level"]
    }


def record_history(user_id, waste_type, points, details=""):
    data = load_data()

    if user_id not in data["users"]:
        return {"error": f"User {user_id} was not found."}

    record = {
        "id": len(data["history"]) + 1,
        "user_id": user_id,
        "waste_type": waste_type,
        "points": points,
        "details": details,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    data["history"].append(record)
    save_data(data)
    return record


# LangChain tools are enabled when an API key is configured.
def build_agent():
    from langchain.agents import create_agent
    from langchain.tools import tool
    from langchain_ollama import ChatOllama

    @tool
    def get_user_profile(user_id: str) -> str:
        """Get a user's name, points, and eco level."""
        return json.dumps(get_profile(user_id))

    @tool
    def award_points(user_id: str, points: int) -> str:
        """Award points to a known user. Use only after a verified activity."""
        if points < 0 or points > 100:
            return json.dumps({"error": "Points must be between 0 and 100."})
        return json.dumps(update_points(user_id, points))

    @tool
    def get_waste_history(user_id: str) -> str:
        """Get the recorded waste submissions for a user."""
        data = load_data()
        records = [
            item for item in data["history"]
            if item["user_id"] == user_id
        ]
        return json.dumps(records)

    @tool
    def get_eco_quests() -> str:
        """List available environmental quests and their point rewards."""
        return json.dumps(load_data()["quests"])

    model = ChatOllama(
        model="qwen2.5:3b",
        temperature=0
    )

    return create_agent(
        model=model,
        tools=[
            get_user_profile,
            award_points,
            get_waste_history,
            get_eco_quests
        ],
        system_prompt=(
            "You are EcoReward, a helpful environmental assistant. "
            "Help users understand recycling, their points, waste history "
            "and eco-quests. Use tools to retrieve current user data. "
            "Never claim an activity was verified unless the application "
            "has actually verified it. Do not award points without a "
            "verified activity and explicit user request."
        )
    )



@app.get("/", response_class=HTMLResponse)
def home():
    html_file = Path(__file__).resolve().parent / "index.html"
    return FileResponse(str(html_file))


@app.get("/api/health")
def health():
    return {
        "status": "running",
        "message": "EcoReward backend is ready",
        "python_ai_agent": "Local Qwen model via Ollama",
    }


@app.get("/api/users")
def users():
    return load_data()["users"]


@app.get("/api/users/{user_id}")
def user_profile(user_id: str):
    result = get_profile(user_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.get("/api/leaderboard")
def leaderboard():
    data = load_data()
    results = [
        {"user_id": user_id, **user}
        for user_id, user in data["users"].items()
    ]
    return sorted(results, key=lambda user: user["points"], reverse=True)


@app.get("/api/history/{user_id}")
def history(user_id: str):
    data = load_data()
    if user_id not in data["users"]:
        raise HTTPException(status_code=404, detail="User not found")
    return [
        item for item in data["history"]
        if item["user_id"] == user_id
    ]


class VerifyWasteRequest(BaseModel):
    submission_id: int
    approved: bool


@app.post("/api/verify-waste")
def verify_waste(request: VerifyWasteRequest):
    data = load_data()

    submission = next(
        (
            item for item in data.get("pending_submissions", [])
            if item.get("id") == request.submission_id
        ),
        None
    )

    if submission is None:
        raise HTTPException(status_code=404, detail="Submission not found")

    if submission.get("status") != "pending":
        raise HTTPException(status_code=409, detail="Submission already reviewed")

    if not request.approved:
        submission["status"] = "rejected"
        save_data(data)
        return {
            "status": "rejected",
            "points_awarded": 0
        }

    user_id = submission["user_id"]
    points = int(submission["suggested_points"])
    user = data["users"].get(user_id)

    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    user["points"] += points
    user["level"] = get_level(user["points"])

    submission["status"] = "verified"
    submission["points_awarded"] = points

    history_item = {
        "id": len(data["history"]) + 1,
        "user_id": user_id,
        "waste_type": submission["waste_type"],
        "points": points,
        "details": submission.get("explanation", ""),
        "submission_id": submission["id"],
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    data["history"].append(history_item)
    save_data(data)

    return {
        "status": "verified",
        "user_id": user_id,
        "points_added": points,
        "total_points": user["points"],
        "level": user["level"]
    }

@app.get("/api/pending-submissions")
def get_pending_submissions():
    data = load_data()
    return [
        item for item in data.get("pending_submissions", [])
        if item.get("status") == "pending"
    ]

@app.get("/api/quests")
def quests():
    return load_data()["quests"]


class AgentRequest(BaseModel):
    user_id: str
    message: str


@app.post("/api/agent/chat")
def agent_chat(request: AgentRequest):
    if request.user_id not in load_data()["users"]:
        raise HTTPException(status_code=404, detail="User not found")

   

    try:
        agent = build_agent()
        result = agent.invoke({
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"The current user ID is {request.user_id}. "
                        f"User message: {request.message}"
                    )
                }
            ]
        })
        answer = result["messages"][-1].content
        return {"mode": "langchain_agent", "reply": answer}
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Agent error: {str(exc)}"
        )


@app.post("/api/analyze-waste")
async def analyze_waste(
    user_id: str,
    image: UploadFile = File(...)
):
    if user_id not in load_data()["users"]:
        raise HTTPException(status_code=404, detail="User not found")

    allowed_types = {"image/jpeg", "image/png", "image/webp"}
    if image.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Upload a JPG, PNG, or WEBP image."
        )

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Image is empty.")
    if len(image_bytes) > 8 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Image must be smaller than 8 MB."
        )

    try:
        from langchain_ollama import ChatOllama
        from langchain_core.messages import HumanMessage

        encoded = base64.b64encode(image_bytes).decode("utf-8")
        mime = image.content_type
        model = ChatOllama(model="qwen2.5vl:3b", temperature=0)

        response = model.invoke([
            HumanMessage(content=[
                {
                    "type": "text",
                    "text": (
                        "Classify the main waste item in this image. "
                        "Reply with JSON only, with keys waste_type, "
                        "recyclable, confidence, points, explanation. "
                        "Use waste_type Plastic, Paper, Metal, Glass, "
                        "E-Waste, Organic, or Other. Set points to 10 "
                        "for E-Waste, 5 for recyclable waste, otherwise 0. "
                        "If uncertain, use Other and 0 points. "
                        "Image classification is not proof of actual recycling. "
                        "Do not award points; points_awarded must remain 0."
                    )
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime};base64,{encoded}"
                    }
                }
            ])
        ])

        raw = response.content
        if isinstance(raw, list):
            raw = "".join(
                part.get("text", "")
                for part in raw if isinstance(part, dict)
            )

        raw = raw.strip().replace("```json", "").replace("```", "").strip()
        json_start = raw.find("{")
        json_end = raw.rfind("}")
        if json_start < 0 or json_end < json_start:
            raise ValueError("AI returned invalid JSON.")

        result = json.loads(raw[json_start:json_end + 1])
        result["status"] = "classified"
        result["user_id"] = user_id



        data = load_data()
        data.setdefault("pending_submissions", [])

        submission_id = max(
            [item.get("id", 0) for item in data["pending_submissions"]],
            default=0
        ) + 1

        waste_type = str(result.get("waste_type", "Other"))
        # Calculate suggested points using application rules, not AI-provided points.
        if waste_type.lower() == "e-waste":
            suggested_points = 10
        elif bool(result.get("recyclable", False)) and waste_type.lower() != "other":
            suggested_points = 5
        else:
            suggested_points = 0

        submission = {
            "id": submission_id,
            "user_id": user_id,
            "waste_type": waste_type,
            "suggested_points": suggested_points,
            "explanation": str(result.get("explanation", "")),
            "status": "pending",
            "points_awarded": 0
        }

        data["pending_submissions"].append(submission)
        save_data(data)
        result["submission_id"] = submission_id
        result["verification_status"] = "pending"
        result["suggested_points"] = suggested_points
        result["points_awarded"] = 0
        result["message"] = (
            "Classification complete. Points have not been awarded "
            "automatically; verification is required."
        )
       
        return result

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Image analysis failed: {str(exc)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)

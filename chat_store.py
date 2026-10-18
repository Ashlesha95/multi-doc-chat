import os
import json
import time
from datetime import datetime

CHAT_DIR = "./chat_history"
os.makedirs(CHAT_DIR, exist_ok=True)


def _path(chat_id: str) -> str:
    return os.path.join(CHAT_DIR, f"{chat_id}.json")


def create_new_chat(docs: list) -> dict:
    chat = {
        "id": f"chat_{int(time.time() * 1000)}",
        "title": "New Chat",
        "created_at": datetime.now().strftime("%b %d, %Y %I:%M %p"),
        "docs": docs,
        "messages": [],
    }
    save_chat(chat)
    return chat


def save_chat(chat: dict):
    with open(_path(chat["id"]), "w") as f:
        json.dump(chat, f, indent=2)


def load_chat(chat_id: str):
    p = _path(chat_id)
    if os.path.exists(p):
        with open(p, "r") as f:
            return json.load(f)
    return None


def get_all_chats() -> list:
    chats = []
    for filename in os.listdir(CHAT_DIR):
        if filename.endswith(".json"):
            with open(os.path.join(CHAT_DIR, filename), "r") as f:
                chats.append(json.load(f))
    chats.sort(key=lambda c: c["id"], reverse=True)
    return chats


def delete_chat(chat_id: str):
    p = _path(chat_id)
    if os.path.exists(p):
        os.remove(p)


def generate_title(first_message: str) -> str:
    title = first_message.strip()
    return title[:40] + "..." if len(title) > 40 else title
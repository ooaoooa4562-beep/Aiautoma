import os
import json
import hmac
import logging
import sqlite3
from contextlib import closing
from typing import Any

import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("telegram-ai-assistant")

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")
AUTO_REPLY = os.getenv("AUTO_REPLY", "false").lower() == "true"
DB_PATH = os.getenv("DB_PATH", "data/assistant.sqlite3")
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "12"))
MAX_REPLY_CHARS = int(os.getenv("MAX_REPLY_CHARS", "1800"))

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-20b:free")
AI_PROVIDER_ORDER = os.getenv("AI_PROVIDER_ORDER", "groq,openrouter").split(",")

SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    """Ты — помощник, который предлагает черновики ответов от имени владельца аккаунта.
Подстраивайся под тон собеседника: будь естественным, дружелюбным и кратким.
Не выдумывай факты о владельце, его планах, местоположении или обещаниях.
Если вопрос требует личного решения, денег, паролей, кодов, документов,
медицинских/юридических выводов или содержит конфликт — вежливо предложи,
что владелец ответит сам. Не проси и не раскрывай пароли, коды входа и секреты.
Не отправляй ссылки, файлы, рекламу или массовые сообщения.
Отвечай на языке собеседника."""
)

app = FastAPI(title="Telegram AI Assistant", version="0.1.0")


def db():
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        telegram_message_id TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS processed (
        message_key TEXT PRIMARY KEY,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.commit()
    return conn


def already_processed(key: str) -> bool:
    with closing(db()) as conn:
        row = conn.execute("SELECT 1 FROM processed WHERE message_key=?", (key,)).fetchone()
        return row is not None


def mark_processed(key: str):
    with closing(db()) as conn:
        conn.execute("INSERT OR IGNORE INTO processed(message_key) VALUES(?)", (key,))
        conn.commit()


def save_message(chat_id: str, role: str, content: str, message_id: str | None = None):
    with closing(db()) as conn:
        conn.execute(
            "INSERT INTO messages(chat_id, role, content, telegram_message_id) VALUES(?,?,?,?)",
            (chat_id, role, content[:5000], message_id),
        )
        conn.execute(
            """DELETE FROM messages WHERE chat_id=? AND id NOT IN
               (SELECT id FROM messages WHERE chat_id=? ORDER BY id DESC LIMIT ?)""",
            (chat_id, chat_id, MAX_HISTORY),
        )
        conn.commit()


def get_history(chat_id: str):
    with closing(db()) as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE chat_id=? ORDER BY id ASC",
            (chat_id,),
        ).fetchall()
    return [{"role": role, "content": content} for role, content in rows]


async def ask_provider(provider: str, messages: list[dict[str, str]]) -> str:
    if provider == "groq":
        key, model = GROQ_API_KEY, GROQ_MODEL
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {key}"}
    elif provider == "openrouter":
        key, model = OPENROUTER_API_KEY, OPENROUTER_MODEL
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {key}",
            "HTTP-Referer": os.getenv("APP_URL", "https://render.com"),
            "X-Title": "Telegram AI Assistant",
        }
    else:
        raise ValueError("Unknown AI provider")

    if not key:
        raise RuntimeError(f"{provider}: API key is not configured")

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 350,
    }
    async with httpx.AsyncClient(timeout=35) as client:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
    answer = data["choices"][0]["message"]["content"]
    if isinstance(answer, list):
        answer = "".join(part.get("text", "") for part in answer if isinstance(part, dict))
    answer = str(answer).strip()
    if not answer:
        raise RuntimeError(f"{provider}: empty response")
    return answer[:MAX_REPLY_CHARS]


async def generate_reply(chat_id: str, incoming: str) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(get_history(chat_id))
    messages.append({"role": "user", "content": incoming})

    errors = []
    for provider in [p.strip().lower() for p in AI_PROVIDER_ORDER if p.strip()]:
        try:
            answer = await ask_provider(provider, messages)
            log.info("AI reply generated with %s", provider)
            return answer
        except Exception as exc:
            log.warning("%s provider failed: %s", provider, exc)
            errors.append(f"{provider}: {exc}")
    raise RuntimeError("All AI providers failed: " + "; ".join(errors))


async def telegram_send(chat_id: int, text: str, business_connection_id: str):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "business_connection_id": business_connection_id,
    }
    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        if not data.get("ok"):
            raise RuntimeError(f"Telegram API error: {data}")
        return data


@app.get("/")
async def root():
    return {"name": "Telegram AI Assistant", "version": "0.1.0", "status": "running"}


@app.get("/health")
async def health():
    return {"ok": True, "auto_reply": AUTO_REPLY}


@app.post("/telegram/webhook")
async def telegram_webhook(request: Request):
    if not BOT_TOKEN or not WEBHOOK_SECRET:
        raise HTTPException(status_code=503, detail="Telegram environment is not configured")

    supplied_secret = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
    if not hmac.compare_digest(supplied_secret, WEBHOOK_SECRET):
        raise HTTPException(status_code=403, detail="Invalid webhook secret")

    update: dict[str, Any] = await request.json()
    message = update.get("business_message")
    if not message:
        return JSONResponse({"ok": True, "ignored": "not a business message"})

    # Only handle ordinary private text messages. Never answer groups, channels,
    # outgoing messages, media-only messages, or messages without a connection ID.
    chat = message.get("chat", {})
    sender = message.get("from", {})
    connection_id = message.get("business_connection_id")
    text_in = message.get("text", "").strip()
    if (
        chat.get("type") != "private"
        or message.get("out")
        or sender.get("is_bot")
        or not connection_id
        or not text_in
    ):
        return JSONResponse({"ok": True, "ignored": "message outside supported scope"})

    chat_id = int(chat["id"])
    message_id = str(message.get("message_id", ""))
    key = f"{connection_id}:{chat_id}:{message_id}"
    if already_processed(key):
        return JSONResponse({"ok": True, "duplicate": True})
    mark_processed(key)

    chat_key = f"{connection_id}:{chat_id}"
    save_message(chat_key, "user", text_in, message_id)

    if not AUTO_REPLY:
        log.info("DRY RUN: received a message from chat %s; no reply sent", chat_id)
        return JSONResponse({"ok": True, "dry_run": True})

    try:
        answer = await generate_reply(chat_key, text_in)
        await telegram_send(chat_id, answer, connection_id)
        save_message(chat_key, "assistant", answer)
        return JSONResponse({"ok": True, "sent": True})
    except Exception:
        log.exception("Could not generate/send reply")
        return JSONResponse({"ok": True, "sent": False, "reason": "AI or Telegram API error"})


@app.post("/admin/set-webhook")
async def set_webhook(request: Request):
    # Protect this setup endpoint with a separate admin key.
    admin_key = os.getenv("ADMIN_KEY", "")
    supplied = request.headers.get("X-Admin-Key", "")
    if not admin_key or not hmac.compare_digest(supplied, admin_key):
        raise HTTPException(status_code=403, detail="Invalid admin key")

    body = await request.json()
    public_url = str(body.get("public_url", "")).rstrip("/")
    if not public_url.startswith("https://"):
        raise HTTPException(status_code=400, detail="public_url must start with https://")

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/setWebhook"
    payload = {
        "url": public_url + "/telegram/webhook",
        "secret_token": WEBHOOK_SECRET,
        "allowed_updates": ["business_message", "business_connection"],
        "drop_pending_updates": False,
    }
    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        return response.json()

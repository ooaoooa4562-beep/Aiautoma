
import os
import logging
import secrets
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Header, HTTPException, Request

# =========================
# CONFIGURATION
# =========================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)

log = logging.getLogger("telegram-assistant")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()

# Your Render URL
PUBLIC_URL = os.getenv(
    "PUBLIC_URL",
    "https://telegram-ai-assistant-1e5s.onrender.com"
).strip().rstrip("/")

WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "").strip()
ADMIN_KEY = os.getenv("ADMIN_KEY", "").strip()

AUTO_REPLY = os.getenv("AUTO_REPLY", "false").lower() == "true"

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "llama-3.3-70b-versatile"
)

OPENROUTER_MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "openai/gpt-4o-mini"
)

SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "Ты дружелюбный личный AI-ассистент. "
    "Отвечай естественно и учитывай контекст переписки."
)

TELEGRAM_API = (
    f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
    if TELEGRAM_TOKEN else ""
)

business_connections = {}
processed_updates = set()


# =========================
# TELEGRAM API
# =========================

async def telegram_call(method: str, payload: dict):
    if not TELEGRAM_API:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is missing")

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"{TELEGRAM_API}/{method}",
            json=payload
        )

        response.raise_for_status()
        data = response.json()

    if not data.get("ok"):
        raise RuntimeError(f"Telegram error: {data}")

    return data


# =========================
# AUTOMATIC WEBHOOK
# =========================

async def register_webhook():
    if not TELEGRAM_TOKEN:
        log.warning("Telegram token is missing")
        return

    if not PUBLIC_URL:
        log.warning("PUBLIC_URL is missing")
        return

    payload = {
        "url": f"{PUBLIC_URL}/telegram/webhook",
        "allowed_updates": [
            "business_connection",
            "business_message"
        ],
        "drop_pending_updates": False
    }

    if WEBHOOK_SECRET:
        payload["secret_token"] = WEBHOOK_SECRET

    try:
        result = await telegram_call("setWebhook", payload)
        log.info(
            "Webhook registered: %s",
            result.get("description", "OK")
        )
    except Exception:
        log.exception("Webhook registration failed")


# =========================
# AI ENGINE
# =========================

async def generate_reply(user_text: str) -> str:

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": user_text[:12000]
        }
    ]

    providers = []

    if GROQ_API_KEY:
        providers.append({
            "name": "Groq",
            "url": "https://api.groq.com/openai/v1/chat/completions",
            "key": GROQ_API_KEY,
            "model": GROQ_MODEL
        })

    if OPENROUTER_API_KEY:
        providers.append({
            "name": "OpenRouter",
            "url": "https://openrouter.ai/api/v1/chat/completions",
            "key": OPENROUTER_API_KEY,
            "model": OPENROUTER_MODEL
        })

    last_error = None

    for provider in providers:

        headers = {
            "Authorization": f"Bearer {provider['key']}",
            "Content-Type": "application/json"
        }

        if provider["name"] == "OpenRouter":
            headers["HTTP-Referer"] = PUBLIC_URL
            headers["X-Title"] = "Telegram AI Assistant"

        try:
            async with httpx.AsyncClient(timeout=45) as client:
                response = await client.post(
                    provider["url"],
                    headers=headers,
                    json={
                        "model": provider["model"],
                        "messages": messages,
                        "temperature": 0.7
                    }
                )

                response.raise_for_status()
                data = response.json()

            answer = data["choices"][0]["message"]["content"]

            if answer and answer.strip():
                log.info("AI response generated with %s", provider["name"])
                return answer.strip()[:4000]

        except Exception as error:
            last_error = error
            log.warning(
                "%s failed: %s",
                provider["name"],
                error
            )

    raise RuntimeError(f"All AI providers failed: {last_error}")


# =========================
# SEND BUSINESS MESSAGE
# =========================

async def send_business_reply(connection_id, chat_id, text):

    chunks = [
        text[i:i + 4000]
        for i in range(0, len(text), 4000)
    ]

    for chunk in chunks:
        await telegram_call(
            "sendMessage",
            {
                "business_connection_id": connection_id,
                "chat_id": chat_id,
                "text": chunk
            }
        )


# =========================
# BUSINESS CONNECTION
# =========================

async def handle_business_connection(connection):

    connection_id = connection.get("id")

    if not connection_id:
        return

    if connection.get("is_enabled"):
        business_connections[connection_id] = connection
        log.info("Business connection enabled")

    else:
        business_connections.pop(connection_id, None)
        log.info("Business connection disabled")


# =========================
# MESSAGE HANDLER
# =========================

async def handle_business_message(message):

    connection_id = message.get("business_connection_id")
    chat = message.get("chat", {})
    chat_id = chat.get("id")

    text = message.get("text") or message.get("caption") or ""

    if not connection_id or not chat_id or not text.strip():
        return

    if not AUTO_REPLY:
        log.info("AUTO_REPLY disabled")
        return

    connection = business_connections.get(connection_id)

    if connection is not None:
        rights = connection.get("rights") or {}

        if not rights.get("can_reply", False):
            log.info("No permission to reply")
            return

    sender = message.get("from", {})

    if sender.get("is_bot"):
        return

    try:
        answer = await generate_reply(text)

        await send_business_reply(
            connection_id,
            chat_id,
            answer
        )

    except Exception:
        log.exception("Message processing failed")


# =========================
# APPLICATION LIFESPAN
# =========================

@asynccontextmanager
async def lifespan(app: FastAPI):

    log.info("NEXUS TELEGRAM AI ONLINE")
    log.info("PUBLIC_URL: %s", PUBLIC_URL)

    await register_webhook()

    yield

    log.info("Application stopped")


# =========================
# FASTAPI
# =========================

app = FastAPI(
    title="Telegram AI Assistant",
    version="0.2.0",
    lifespan=lifespan
)


@app.get("/")
async def root():

    return {
        "service": "Telegram AI Assistant",
        "status": "online",
        "version": "0.2.0"
    }


@app.get("/health")
async def health():

    return {
        "ok": True,
        "auto_reply": AUTO_REPLY,
        "webhook_configured": bool(
            TELEGRAM_TOKEN and PUBLIC_URL
        ),
        "ai_provider_configured": bool(
            GROQ_API_KEY or OPENROUTER_API_KEY
        )
    }


# =========================
# TELEGRAM WEBHOOK
# =========================

@app.post("/telegram/webhook")
async def telegram_webhook(
    request: Request,
    x_telegram_bot_api_secret_token: str | None = Header(default=None)
):

    if WEBHOOK_SECRET and not secrets.compare_digest(
        x_telegram_bot_api_secret_token or "",
        WEBHOOK_SECRET
    ):
        raise HTTPException(
            status_code=403,
            detail="Invalid webhook secret"
        )

    update = await request.json()

    update_id = update.get("update_id")

    if isinstance(update_id, int):

        if update_id in processed_updates:
            return {"ok": True, "duplicate": True}

        processed_updates.add(update_id)

        if len(processed_updates) > 5000:
            processed_updates.clear()
            processed_updates.add(update_id)

    if "business_connection" in update:

        await handle_business_connection(
            update["business_connection"]
        )

    elif "business_message" in update:

        await handle_business_message(
            update["business_message"]
        )

    return {"ok": True}


# =========================
# ADMIN: SET WEBHOOK
# =========================

@app.post("/admin/set-webhook")
async def admin_set_webhook(
    x_admin_key: str | None = Header(default=None)
):

    if not ADMIN_KEY:
        raise HTTPException(
            status_code=503,
            detail="ADMIN_KEY is not configured"
        )

    if not secrets.compare_digest(
        x_admin_key or "",
        ADMIN_KEY
    ):
        raise HTTPException(
            status_code=403,
            detail="Invalid admin key"
        )

    await register_webhook()

    return {
        "ok": True,
        "webhook_url": f"{PUBLIC_URL}/telegram/webhook"
    }

import uuid, requests, time, os
from dotenv import load_dotenv

load_dotenv()

QWEN_TOKEN = os.environ.get("QWEN_TOKEN", "")
QWEN_COOKIES = os.environ.get("QWEN_COOKIES", "")
QWEN_BASE_URL = "https://chat.qwen.ai"
DEFAULT_QWEN_MODEL = "qwen3.7-plus"

headers = {
            "Authorization": f"Bearer {QWEN_TOKEN}" if QWEN_TOKEN else "",
            "X-Request-Id": str(uuid.uuid4()),
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept": "*/*",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Referer": f"{QWEN_BASE_URL}/",
            "Origin": QWEN_BASE_URL,
            "Connection": "keep-alive",
            "Content-Type": "application/json",
            "sec-ch-ua": '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "cookies": QWEN_COOKIES
        }
cid = '4f782e51-cc30-4717-a40c-6f8ebca89818'
#"chat_id": "601c032d-15a7-4fd7-8778-9b4b03ee13b6",
data = {
    "stream": True,
    "version": "2.1",
    "incremental_output": True,
    "chat_id": cid,
    "chat_mode": "normal",
    "model": "qwen3.7-plus",
    "parent_id": None,
    "messages": [
        {
            "fid": "784577c5-8a25-4d8e-ba28-70dcb58d7e47",
            "parentId": None,
            "childrenIds": [
                "af325b6e-e9f6-475e-bbda-f64e7a58d66a"
            ],
            "role": "user",
            "content": "Explain Quantum Physics",
            "user_action": "chat",
            "files": [],
            "timestamp": int(time.time()),
            "models": [
                "qwen3.7-plus"
            ],
            "chat_type": "t2t",
            "feature_config": {
                "thinking_enabled": True,
                "output_schema": "phase",
                "research_mode": "normal",
                "auto_thinking": True,
                "thinking_mode": "Auto",
                "thinking_format": "summary",
                "auto_search": True
            },
            "extra": {
                "meta": {
                    "subChatType": "t2t"
                }
            },
            "sub_chat_type": "t2t"
        }
    ],
    "timestamp": int(time.time())
}

print(requests.post(f"{QWEN_BASE_URL}/api/v2/chat/completions?chat_id={cid}", headers=headers, json=data).text)

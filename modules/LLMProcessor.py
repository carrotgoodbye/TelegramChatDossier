import json
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import requests
import time
import logging

from utils.prompts import Prompts

logger = logging.getLogger("LLMProcessor")


class LLMProcessor:
    MODEL_CONTEXT_SIZES = {
        "llama3.1": 131072, "llama3.2": 131072, "mistral": 32768,
        "mistral-nemo": 131072, "qwen2.5": 131072, "qwen2": 32768,
        "gemma2": 8192, "phi3": 131072, "command-r": 131072,
        "deepseek-coder-v2": 131072, "mixtral": 32768, "qwen3": 40960,
        "default": 8192,
    }

    def __init__(self, ollama_url: str, model: str):
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model
        self.max_retries = 3

        base_name = self.model.split(":")[0].lower()
        self.context_size = self.MODEL_CONTEXT_SIZES.get(base_name, self.MODEL_CONTEXT_SIZES["default"])

        self.system_prompt_tokens = 2000
        self.response_reserve = 4000
        self.buffer_tokens = 500
        self.max_chunk_chars = int(
            (self.context_size - self.system_prompt_tokens - self.response_reserve - self.buffer_tokens) / 0.75
        )

        logger.info(f"Model: {self.model}, Context: {self.context_size}, Max chunk: ~{self.max_chunk_chars} chars")
        self._check_connection()
        self._check_model()

    def _check_connection(self):
        try:
            resp = requests.get(f"{self.ollama_url}/api/tags", timeout=10)
            resp.raise_for_status()
            logger.info("Connected to Ollama")
        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"Cannot connect to Ollama at {self.ollama_url}. Run: ollama serve")

    def _check_model(self):
        try:
            resp = requests.get(f"{self.ollama_url}/api/tags", timeout=10)
            models = resp.json().get("models", [])
            model_names = [m.get("name", m.get("model", "")) for m in models]
            if self.model not in model_names:
                logger.info(f"Pulling {self.model}...")
                requests.post(f"{self.ollama_url}/api/pull",
                    json={"name": self.model, "stream": False}, timeout=600)
        except Exception as e:
            logger.error(str(e))

    def _estimate_tokens(self, text: str) -> int:
        return int(len(text) / 1.5) + 100

    def _truncate_messages(self, messages: List[Dict], max_tokens: int) -> List[Dict]:
        system_msg = messages[0]
        user_msg = messages[1]
        system_tokens = self._estimate_tokens(system_msg["content"])
        available = max_tokens - system_tokens - self.response_reserve - self.buffer_tokens

        user_text = user_msg["content"]
        if self._estimate_tokens(user_text) > available:
            max_chars = int(available * 1.5)
            if max_chars < 1000:
                truncated = user_text[-max_chars:]
            else:
                half = max_chars // 2
                truncated = user_text[:half] + "\n\n[... truncated ...]\n\n" + user_text[-half:]
            logger.debug(f"Truncated: {len(user_text)} -> ~{len(truncated)} chars")
            messages[1]["content"] = truncated
        return messages

    def _call(self, messages: List[Dict], temperature: float = 0.1) -> str:
        messages = self._truncate_messages(messages, self.context_size)
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": self.response_reserve,
                "top_p": 0.9,
                "num_ctx": min(self.context_size, 65536),
            }
        }

        for attempt in range(self.max_retries):
            try:
                resp = requests.post(f"{self.ollama_url}/api/chat", json=payload, timeout=600)

                if resp.status_code == 500:
                    error_text = resp.text
                    logger.error(f"500 Error (attempt {attempt + 1}): {error_text[:200]}")
                    if "context length" in error_text.lower() or "token" in error_text.lower():
                        self.response_reserve = int(self.response_reserve * 0.7)
                        payload["options"]["num_predict"] = self.response_reserve
                        messages[1]["content"] = messages[1]["content"][:int(len(messages[1]["content"]) * 0.7)]
                        time.sleep(2)
                        continue
                    time.sleep(5 * (attempt + 1))
                    continue

                resp.raise_for_status()
                data = resp.json()
                if "error" in data:
                    raise RuntimeError(f"Ollama error: {data['error']}")
                content = data.get("message", {}).get("content", "")
                if not content or len(content.strip()) < 50:
                    time.sleep(2)
                    continue
                return content
            except requests.exceptions.Timeout:
                time.sleep(10 * (attempt + 1))
            except requests.exceptions.RequestException as e:
                if attempt == self.max_retries - 1:
                    raise e
                time.sleep(5 * (attempt + 1))
        return "{}"

    def extract_from_chunk(self, chunk_text: str, chunk_meta: Dict) -> Dict[str, Any]:
        system_prompt = Prompts.SYSTEM_PROMPT
        user_prompt = Prompts.user_prompt(chunk_text, chunk_meta)

        raw_response = self._call([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ], temperature=0.1)

        result = self._parse_json_response(raw_response)

        # Если парсинг вернул пустой fallback — логируем сырой ответ для отладки
        if not result.get('entities') and not result.get('facts'):
            debug_path = Path("llm_debug_responses")
            debug_path.mkdir(exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            (debug_path / f"chunk_{ts}_raw.txt").write_text(raw_response, encoding='utf-8')

        return result

    def process_with_prompt(self, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> Dict[str, Any]:
        """Универсальный метод для пост-обработки с кастомным промптом."""
        raw_response = self._call([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ], temperature=temperature)
        return self._parse_json_response(raw_response)

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
            # Если LLM вернул список вместо dict
            if isinstance(parsed, list):
                return {"entities": parsed, "relations": [], "cognitive_patterns": [],
                        "personality_dimensions": [], "facts": [], "communication_styles": [], "social_graph": []}
        except json.JSONDecodeError:
            pass

        # Попытка найти JSON-объект внутри текста
        match = re.search(r'\{[\s\S]*}', text)
        if match:
            try:
                parsed = json.loads(match.group())
                if isinstance(parsed, dict):
                    return parsed
            except:
                pass

        return {
            "entities": [], "relations": [], "cognitive_patterns": [],
            "personality_dimensions": [], "facts": [], "communication_styles": [], "social_graph": []
        }

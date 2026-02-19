import requests
from utils.logger_setup import getlogger
from flask import request

logger = getlogger("llm_rag")


class RagLLM:
    # Default API URL
    api_url = "http://ai-rag-01:9000/ask_streamed"

    def _get_user_ip(self):
        """
        Extract the user's IP address from the Flask request headers.
        Falls back to request.remote_addr or 'unknown'.
        """
        if not request:
            return "unknown"
        x_forwarded_for = request.headers.get("X-Forwarded-For")
        if x_forwarded_for:
            return x_forwarded_for.split(",")[0].strip()
        return request.remote_addr or "unknown"

    def stream(self, question: str):
        """
        Stream a RAG response chunk-by-chunk to the GUI.
        Logs the start, ongoing streaming, and end.
        """
        user_ip = self._get_user_ip()
        logger.info(f"User message from {user_ip}: {question}")

        payload = {"question": question}
        logger.info(f"[RAG] Payload: {payload}")

        full_response = ""

        try:
            # Start streaming request to the RAG API
            with requests.post(self.api_url, json=payload, stream=True) as response:
                response.raise_for_status()
                # Log response metadata
                logger.info(f"[RAG] Response status: {response.status_code} {response.reason}, elapsed: {response.elapsed}")
                logger.info(f"[RAG] Response headers: {response.headers}")
                logger.info("[RAG] Streaming started and ongoing...")

                # Stream the content in small chunks
                for chunk_bytes in response.iter_content(chunk_size=64, decode_unicode=True):
                    if not chunk_bytes:
                        continue
                    chunk = chunk_bytes
                    full_response += chunk
                    yield chunk  # Send each chunk to the GUI

        except requests.RequestException as e:
            logger.error(f"[RAG] Request failed: {e}")
            yield f"[Error fetching RAG response: {e}]"

        # Log streaming finished
        logger.info("[RAG] Streaming finished.")
        # Log full assistant response just once (to avoid log flooding)
        logger.info(f"[RAG] Full assistant response:\n{full_response}")

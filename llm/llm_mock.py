import time
from utils.logger_setup import getlogger

logger = getlogger("llm_mock")

class MockLLM:

    def stream(self, prompt: str):
        logger.info(f"[Mock] Prompt ({len(prompt)} chars): {prompt!r}")

        fake_answer = f"[Mock] You said: {prompt}\nThis is a mock response."

        for token in fake_answer.split():
            yield token + " "
            time.sleep(0.03)

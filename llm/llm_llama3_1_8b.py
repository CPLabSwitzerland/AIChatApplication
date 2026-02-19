import requests
import json
from utils.logger_setup import getlogger

# Set up a dedicated logger for this module
logger = getlogger("llm_llama3_1_8b")

# Configuration constants
LLAMA_API_URL = "http://ai-llm-01:8082/v1/completions"  # TinyLlama API endpoint
MAX_TOKENS = 450      # Limit response length for safety
N_CTX = 2048          # Model context window
TEMPERATURE = 0.1     # Controls randomness of the output (lower = more deterministic)
MODEL_NAME = "llama-3.1-8b-instruct.Q4_K_M.gguf"  # Model to use
STOP_SEQUENCE = "\n"  # Stop generation when a newline is encountered

class Llama3_1_8bLLM:
    """
    Llama 3.1_8b LLM integration class.

    Responsibilities:
    - Construct prompts for questions.
    - Send prompts to Llama 3.1_8b API with streaming enabled.
    - Yield model response in chunks.
    - Log the prompt, each chunk, and the final full response for debugging.
    """

    def build_prompt(self, question: str) -> str:
        """
        Build a prompt for Llama 3.1_8b to answer a single question concisely.

        Rules enforced in the prompt:
        - Answer in one sentence only.
        - Do not add extra text or commentary.
        - Do not repeat the question.
        - Do not generate any new questions.

        Args:
            question (str): The user's input question.

        Returns:
            str: The formatted prompt string.
        """
        return (
        "SYSTEM: You are a helpful assistant.\n"
        "SYSTEM: Answer questions in exactly one sentence. "
        "Your answer should be concise but informative, providing key context if relevant. "
        "Do not describe yourself, do not repeat phrases, do not add extra information beyond the topic, and do not ask questions. "
        "End your answer after the first period.\n\n"
        f"USER: {question}\n"
        "ASSISTANT: (one informative sentence only)"
        )
    def stream(self, prompt: str):
        """
        Send the prompt to Llama 3.1_8b API and stream back the response.

        The function:
        - Logs the full prompt for debugging.
        - Sends a POST request with streaming enabled.
        - Iterates over streamed lines, parsing JSON for text chunks.
        - Handles stop sequences to end generation cleanly.
        - Logs each chunk and the full assistant response.

        Args:
            prompt (str): The user's question (raw input).

        Yields:
            str: Each chunk of the assistant's response as it arrives.
        """
        full_prompt = self.build_prompt(prompt)

        # Log the prompt being sent for debugging
        logger.info(f"[Llama3_1_8b] Sending full prompt ({len(full_prompt)} chars): {full_prompt!r}")

        # Build payload for the API
        payload = {
            "model": MODEL_NAME,
            "prompt": full_prompt,
            "max_tokens": MAX_TOKENS,
            "n_ctx": N_CTX,
            "temperature": TEMPERATURE,
            "stop": STOP_SEQUENCE,
            "stream": True
        }

        assistant_response = ""  # Collect the full response for logging

        try:
            # Send the request to the TinyLlama API
            with requests.post(LLAMA_API_URL, json=payload, stream=True) as response:
                response.raise_for_status()

                # Iterate over the streamed lines from the API
                for line in response.iter_lines(decode_unicode=True):
                    if not line or line.strip() == "[DONE]":
                        continue  # Skip empty lines and the '[DONE]' signal

                    if line.startswith("data: "):
                        data_str = line[len("data: "):]
                        try:
                            # Parse the JSON data from the stream
                            data_json = json.loads(data_str)
                            for choice in data_json.get("choices", []):
                                text = choice.get("text", "")
                                if text:
                                    # Stop generation if the stop sequence is detected
                                    if STOP_SEQUENCE in text:
                                        text = text.split(STOP_SEQUENCE)[0]
                                        assistant_response += text
                                        logger.info(f"[Llama3_1_8b] chunk (stopped): {text!r}")
                                        yield text
                                        logger.info(f"[Llama3_1_8b] Streaming finished.")
                                        return
                                    assistant_response += text
                                    logger.info(f"[Llama3_1_8b] chunk: {text!r}")
                                    yield text
                        except json.JSONDecodeError:
                            logger.warning(f"[Llama3_1_8b] Could not parse JSON: {data_str!r}")

        except requests.RequestException as e:
            logger.error(f"[Llama3_1_8b] Request failed: {e}")

        # Log the full response at the end if available
        if assistant_response:
            logger.info(f"[Llama3_1_8b] Full assistant response: {assistant_response!r}")

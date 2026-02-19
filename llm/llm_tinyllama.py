import requests
import json
from utils.logger_setup import getlogger

# Set up a dedicated logger for this module
logger = getlogger("llm_tinyllama")

# Configuration constants
LLAMA_API_URL = "http://ai-llm-01:8081/v1/completions"  # TinyLlama API endpoint
MAX_TOKENS = 250      # Limit response length for safety
N_CTX = 2048          # Model context window
TEMPERATURE = 0.1     # Controls randomness of the output (lower = more deterministic)
MODEL_NAME = "tinylama-rust-q4_k_m.gguf"  # Model to use
STOP_SEQUENCE = "\n"  # Stop generation when a newline is encountered

class TinyLlamaLLM:
    """
    TinyLlama LLM integration class.

    Responsibilities:
    - Construct prompts for questions.
    - Send prompts to TinyLlama API with streaming enabled.
    - Yield model response in chunks.
    - Log the prompt, each chunk, and the final full response for debugging.
    """

    def build_prompt(self, question: str) -> str:
        """
        Build a prompt for TinyLlama to answer a single question concisely.

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
            "\nYou are a helpful assistant.\n"
            "Answer the following question in exactly one sentence only. "
            "Your sentence should be concise but informative, providing key context if relevant. "
            "Do not describe yourself, do not repeat the question, do not ask questions, and do not write more than one period. "
            "After the first period, stop writing immediately.\n\n"
            f"Question: {question}\n"
            "Answer (one informative sentence only):"
        )

    def stream(self, prompt: str):
        """
        Send the prompt to TinyLlama API and stream back the response.

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
        logger.info(f"[TinyLlama] Sending full prompt ({len(full_prompt)} chars): {full_prompt!r}")

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
                                        logger.info(f"[TinyLlama] chunk (stopped): {text!r}")
                                        yield text
                                        logger.info(f"[TinyLlama] Streaming finished.")
                                        return
                                    assistant_response += text
                                    logger.info(f"[TinyLlama] chunk: {text!r}")
                                    yield text
                        except json.JSONDecodeError:
                            logger.warning(f"[TinyLlama] Could not parse JSON: {data_str!r}")

        except requests.RequestException as e:
            logger.error(f"[TinyLlama] Request failed: {e}")

        # Log the full response at the end if available
        if assistant_response:
            logger.info(f"[TinyLlama] Full assistant response: {assistant_response!r}")

import os
import ollama
from src.server import CodeAutocompleteEnv
from src.models import CodeAction
from dotenv import load_dotenv

import logging 

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

load_dotenv()


MODEL_NAME = os.getenv("MODEL_NAME", "llama3")
client = ollama.Client(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
env = CodeAutocompleteEnv()

def get_completion(code_context: str):
    """
    Standard entry point for agentic completion.
    Steps through OpenEnv to get Reward.
    """
    # 1. Reset Env with IDE Context
    obs = env.reset(code_context=code_context)
    
    # 2. Get Suggestion from Ollama
    prompt = f"Complete this Python code. Provide ONLY the next few characters or line:\n{obs.code_context}"
    response = client.chat(model=MODEL_NAME, messages=[{'role': 'user', 'content': prompt}])
    suggestion = response['message']['content']
    
    # 3. Execute Step in OpenEnv to calculate RL Reward
    action = CodeAction(completion=suggestion)
    _, reward, _, _ = env.step(action)
    
    logger.info(f"Reward: {reward}")
    logger.info(f"Suggestion: {suggestion}")
    return {
        "completion": suggestion,
        "reward": reward
    }

if __name__ == "__main__":
    # CLI Testing mode
    print(get_completion("import os\ndef list_files():\n    "))
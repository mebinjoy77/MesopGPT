import mesop as me
import mesop.labs as ml
from azure_functions import AzureLLM
from dotenv import load_dotenv
import os
from enum import Enum

# Load environment variables from a .env file
load_dotenv()

# Retrieve Azure OpenAI service credentials from environment variables
azure_deployment_name = os.getenv("AZURE_OPENAI_MODEL")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_key = os.getenv("AZURE_OPENAI_KEY")
azure_version = os.getenv("AZURE_OPENAI_VERSION")

# Initialize the AzureLLM model with the retrieved credentials
model = AzureLLM(
    azure_endpoint=azure_endpoint,
    azure_key=azure_key,
    azure_deployment_name=azure_deployment_name,
    azure_version=azure_version
)

@me.page(path="/chat", title="Azure AI Chat")
def chatpage():
    """
    Define the chat page for the web application.
    """
    ml.chat(response_func, title="Azure AI Chat", bot_user="Azure AI")

def response_func(user_query: str, history: list[ml.ChatMessage]) -> str:
    """
    Generate a response for the chat page using the AzureLLM model.

    Parameters:
    user_query (str): The query from the user.
    history (list[ml.ChatMessage]): The chat history.

    Returns:
    str: The generated response from the AzureLLM model.
    """
    return model.generate(user_query)

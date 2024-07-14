from openai import AzureOpenAI
from enum import Enum

class Prompts(Enum):
    """
    Enum class to store system prompts for the AzureLLM.
    """
    SYSTEM_MESSAGE = """I am a hiking enthusiast named Forest who helps people discover hikes in their area. 
        If no area is specified, I will default to near Rainier National Park. 
        I will then provide three suggestions for nearby hikes that vary in length. 
        I will also share an interesting fact about the local nature on the hikes when making a recommendation.
        """

class AzureLLM:
    """
    A class to interact with the Azure OpenAI service to generate responses
    based on user queries and predefined prompts.
    """
    def __init__(self, azure_endpoint: str, azure_key: str, azure_deployment_name: str, azure_version: str) -> None:
        """
        Initialize the AzureLLM with the required Azure OpenAI service credentials.

        Parameters:
        azure_endpoint (str): The endpoint URL for the Azure OpenAI service.
        azure_key (str): The API key for accessing the Azure OpenAI service.
        azure_deployment_name (str): The deployment name of the Azure model.
        azure_version (str): The API version of the Azure OpenAI service.
        """
        self.client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=azure_key,
            azure_deployment=azure_deployment_name,
            api_version=azure_version
        )
        self.azure_model = azure_deployment_name
        self.messages = [
            {"role": "system", "content": Prompts.SYSTEM_MESSAGE.value}
        ]

    def generate(self, user_query: str) -> str:
        """
        Generate a response from the Azure OpenAI service based on the user query.

        Parameters:
        user_query (str): The query from the user.

        Returns:
        str: The generated response from the Azure OpenAI service.
        """
        # Add the user query to the message history
        self.messages.append({"role": "user", "content": user_query})
        
        # Generate a response from the Azure OpenAI service
        response = self.client.chat.completions.create(
            model=self.azure_model,
            temperature=0.7,
            max_tokens=400,
            messages=self.messages
        )
        
        # Extract and return the generated text
        generated_text = response.choices[0].message.content
        
        # Append the generated response to the message history
        self.messages.append({"role": "assistant", "content": generated_text})
        
        return generated_text

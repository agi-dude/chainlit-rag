import ollama
import openai
from openai import AzureOpenAI
import anthropic


class Client:
    def __init__(self, provider: str, api_key: str, model: str, host: str = None):
        self.client = None
        self.provider = provider
        self.host = host
        self.api_key = api_key
        self.model = model

        self.initialize_client()

    def initialize_client(self):
        if self.provider == 'openai':
            self.client = openai.Client(api_key=self.api_key, base_url=self.host)

        elif self.provider == 'anthropic':
            self.client = anthropic.Client(api_key=self.api_key)

        elif self.provider == 'azure':
            self.client = AzureOpenAI(azure_endpoint=self.host, api_version="2024-02-01", api_key=self.api_key)

        elif self.provider == 'ollama':
            self.client = ollama.Client(self.host)

    def chat(self, messages):
        if self.provider == 'openai':
            response = self.client.chat.completions.create(
                messages=messages,
                model=self.model,
            )
            return response.choices[0].message.content

        elif self.provider == 'anthropic':
            message = self.client.messages.create(
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": "Hello, Claude",
                    }
                ],
                model=self.model,
            )
            return message.content
        
        elif self.provider == 'azure':
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )
            return completion.choices[0].message.content

        elif self.provider == 'ollama':
            response = self.client.chat(model=self.model, messages=messages)
            return response['message']['content']

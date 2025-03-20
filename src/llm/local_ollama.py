import requests

class LocalOllamaModel:
    def __init__(self, model_name: str, base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url

    def generate(self, prompt) -> str:
        # Prepare the request payload
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "api_type": "ollama"
        }

        # Send the request to the local Ollama model server
        response = requests.post(f"{self.base_url}/generate", json=payload)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the response JSON
            response_data = response.json()
            return response_data.get("generated_text", "No response text found")
        else:
            # Handle error response
            return f"Error: {response.status_code} - {response.text}"

    def invoke(self, prompt) -> str:
        # Call the generate method
        return self.generate(prompt)

    def __repr__(self):
        return f"LocalOllamaModel(model_name={self.model_name}, base_url={self.base_url})"
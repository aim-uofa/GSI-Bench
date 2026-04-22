import base64
import time
import json
import os
from PIL import Image
from io import BytesIO
import traceback
from openai import OpenAI


class GPT:
    """
    Simple interface for interacting with GPT-4O model using OpenAI client
    """

    def __init__(
            self,
            api_key,
            model="gpt-4o",
            max_retries=3,
            base_url="https://api.openai.com/v1"
    ):
        """
        Args:
            api_key (str): Key to use for querying GPT
            model (str): GPT model to use
            max_retries (int): The maximum number of retries to prompt GPT when receiving server error
            base_url (str): Base URL for the API endpoint
        """
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.base_url = base_url

        # Initialize OpenAI client
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )

    def __call__(self, messages, verbose=False, **kwargs):
        """
        Queries GPT using the desired messages

        Args:
            messages (list): Messages to pass to GPT
            verbose (bool): Whether to be verbose as GPT is being queried
            **kwargs: Additional parameters like temperature, max_tokens, etc.

        Returns:
            None or str: Raw outputted GPT response if valid, else None
        """
        attempts = 0
        while attempts < self.max_retries:
            try:
                if verbose:
                    print(f"Querying GPT-{self.model} API...")

                # Set default parameters
                params = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0.1,
                    "max_tokens": 300,
                }
                # Override with any provided kwargs
                params.update(kwargs)

                response = self.client.chat.completions.create(**params)

                if verbose:
                    print(f"Finished querying GPT-{self.model}.")

                return response.choices[0].message.content

            except Exception as e:
                attempts += 1
                print(f"Error querying GPT-{self.model} API (attempt {attempts}): {e}")
                if attempts < self.max_retries:
                    print(f"Retrying in 5 seconds...")
                    time.sleep(5)
                else:
                    print(f"Failed to query GPT-{self.model} API after {self.max_retries} attempts.")
                    traceback.print_exc()
                    return None
        return None

    def generate_task_info(self, task_name: str, table_type: str, verbose=False) -> dict | None:
        """
        Convert task instructions into detailed task information
        
        Args:
            task_name (str): Task name (e.g., "Organize books and magazines")
            table_type (str): Table type (e.g., "Nightstand", "TV stand", "side table")
            verbose (bool): Whether to output detailed information
            
        Returns:
            dict: Parsed task information result, None if failed
        """
        # Format prompt using the imported prompt template
        from .prompt import TASK_INFO_PROMPT
        formatted_prompt = TASK_INFO_PROMPT.format(task_name=task_name, table_type=table_type)
        
        messages = [
            {
                "role": "user",
                "content": formatted_prompt
            }
        ]
        
        response = self(
            messages=messages,
            verbose=verbose,
            response_format={"type": "json_object"},
            max_tokens=800,
            temperature=0.2
        )
        
        if response is None:
            return None
            
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            print(f"Failed to parse GPT response: {e}")
            print(f"Raw response: {response[:200]}{'...' if len(response) > 200 else ''}")
            return None

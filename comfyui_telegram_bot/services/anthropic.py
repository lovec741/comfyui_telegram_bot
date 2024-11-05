import anthropic
from typing import Optional
import re

from ..prompt_enhance import PromptEnhanceService

class AnthopicService(PromptEnhanceService):
    def __init__(self, config):
        super().__init__(config)
        self.client = anthropic.Anthropic(api_key=self.config.api_key)

    def _call_api(self, user_prompt: str, system_prompt: str) -> str:
        message = self.client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1000,
            temperature=0,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_prompt
                        }
                    ]
                }
            ]
        )
        return message.content.pop().text

    def enhance_prompt(self, user_prompt: str, pe_type: str) -> str:
        type_config = self.config.types.get(pe_type)
        if type_config is None:
            raise Exception(f"Invalid prompt enhancement type '{pe_type}'")
        system_prompt = type_config.system_prompt
        retries = self.config.retries
        for attempt in range(retries):
            try:
                prompt = self._call_api(user_prompt, system_prompt)
                prompt_match = re.search(r'<prompt>((.|\n)*?)</prompt>', prompt, re.DOTALL)
                if prompt_match:
                    extracted_prompt = prompt_match.group(1).strip()
                    print("Enhanced prompt:", extracted_prompt)
                    return extracted_prompt
                else:
                    print(f"Attempt {attempt + 1}: No prompt found between <prompt> tags.")
            except Exception as e:
                if attempt == retries - 1:
                    print(f"Failed to create prompt after {retries} attempts. Error: {str(e)}")
                else:
                    print(f"Attempt {attempt + 1} failed. Retrying...")
        else:
            print("Failed to create a valid prompt after all attempts.")
            raise Exception(f"Failed to generate enhanced prompt, please try again later.")
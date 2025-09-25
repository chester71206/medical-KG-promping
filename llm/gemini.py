import os
from langchain_core.language_models.llms import LLM
from typing import Any, List, Mapping, Optional
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
import json
import requests
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("gemini_api_key_upgrade")


class Gemini(LLM):
    API_KEY: str

    @property
    def _llm_type(self) -> str:
        return "Gemini"

    def _call(
        self,
        prompt: str,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        #gemini-1.5-flash
        #gemma-3-1b
        #url = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent?key={self.API_KEY}'
        url = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.API_KEY}'
        #url = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={self.API_KEY}'
        headers = {'Content-Type': 'application/json'}
        data = {"contents": 
                [{
                    "parts": [{"text": prompt}]
                }]
        }
        response = requests.post(url, headers=headers, json=data)

        
        try:
            output = json.dumps(response.json()["candidates"][0]["content"]["parts"][0]["text"], 
                                indent=4, 
                                ensure_ascii=False)

        except KeyError:
            try:
                code = json.dumps(response.json()["candidates"][0]["finishReason"],
                                indent=4, 
                                ensure_ascii=False)
                
            except KeyError:
                code = json.dumps(response.json()["error"]["code"],
                                    indent=4, 
                                    ensure_ascii=False)
            


            if code == "429":
                output = "code429"

            elif code == "SAFETY":
                output = "SAFETY"
            
            elif code == "RECITATION":
                output = "RECITATION"
            
            else:
                # output = json.dumps(response.json()["error"],
                #                 indent=4, 
                #                 ensure_ascii=False)
                output = code
                print("test:", output)

        return output
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"API_KEY": self.API_KEY}
    



# chat = Gemini(API_KEY=api_key)
# prompt = "美國一個的天氣如何？"
# print(chat(prompt))

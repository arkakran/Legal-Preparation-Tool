import json
import requests
from typing import List, Dict, Optional

class GroqClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.groq.com/openai/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def chat_completions_create(self, messages: List[Dict], model: str, temperature: float = 0.15, max_tokens: int = 1400):
        url = f"{self.base_url}/chat/completions"
        data = {
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            print(f"Making API request to: {url}")
            response = requests.post(url, headers=self.headers, json=data, timeout=60)
            print(f"Response status: {response.status_code}")
            
            if response.status_code != 200:
                print(f"API Error: {response.text}")
                return None
                
            result = response.json()
            print(f"API Response received successfully")
            return result
            
        except requests.exceptions.Timeout:
            print("API request timed out")
            return None
        except Exception as e:
            print(f"API request failed: {e}")
            return None

class LLMAnalyzer:
    def __init__(self, api_key: str):
        self.client = GroqClient(api_key)
        self.model = "llama-3.3-70b-versatile"  # Changed model name - this one is more reliable

    def _extract_json_from_text(self, text: str) -> Optional[List[Dict]]:
        if not text:
            return None
            
        # Clean the text first
        text = text.strip()
        
        # Try direct JSON parsing first
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return data
            return None
        except json.JSONDecodeError:
            pass
        
        # Find JSON array using regex
        import re
        
        # Look for array patterns
        patterns = [
            r'\[[\s\S]*\]',  # Any content between square brackets
            r'```json\s*(\[[\s\S]*?\])\s*```',  # JSON in code blocks
            r'```\s*(\[[\s\S]*?\])\s*```',  # JSON in any code blocks
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                for match in matches:
                    try:
                        # Clean up the match
                        cleaned = match.strip()
                        if not cleaned.startswith('['):
                            continue
                        data = json.loads(cleaned)
                        if isinstance(data, list):
                            return data
                    except json.JSONDecodeError:
                        continue
        
        print(f"Could not extract JSON from LLM response: {text[:500]}...")
        return None

    def analyze_points(self, top_text_block: str) -> List[Dict]:
        if not self.client or not top_text_block:
            print("No client or text block available for LLM analysis")
            return []

        # Truncate text to ensure we don't exceed token limits
        text_block = top_text_block[:25000]  # Reduced size for safety
        
        prompt = f"""Extract the most important legal points from this text as a JSON array. Each point should have exactly these fields:
- "summary": Brief 1-2 sentence description
- "stance": "for", "against", or "neutral" 
- "importance": number from 1-10
- "category": "argument", "precedent", "statute", "evidence", or "procedure"
- "quote": short exact quote from the text (8-25 words)

Return exactly 8-12 points. Output only valid JSON array, no other text.

TEXT:
{text_block}"""

        try:
            print("Sending request to LLM...")
            response = self.client.chat_completions_create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.1,
                max_tokens=2000,
            )
            
            if not response or "choices" not in response:
                print("Invalid LLM response structure")
                return []
                
            content = response["choices"][0]["message"]["content"]
            print(f"LLM response received: {len(content)} characters")
            
            data = self._extract_json_from_text(content)
            if isinstance(data, list) and len(data) > 0:
                print(f"Successfully extracted {len(data)} LLM points")
                return data[:12]  # Limit to 12 points
            else:
                print("No valid JSON array extracted from LLM response")
                return []
                
        except Exception as e:
            print(f"LLM analysis error: {e}")
            return []
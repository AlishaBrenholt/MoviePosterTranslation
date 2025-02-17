import requests
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("API_KEY")
print(f"API Key: {api_key}")

movie_id = 496243
url = f"https://api.themoviedb.org/3/movie/{movie_id}/images?include_image_language=en%2Ckor"
headers = {
    "accept": "application/json",
    "Authorization": f"Bearer {api_key}"
}

response = requests.get(url, headers=headers)

print(response.text)
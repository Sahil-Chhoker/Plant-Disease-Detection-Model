import requests
import json
import dotenv
import os
import base64

dotenv.load_dotenv()

url = "https://plant.id/api/v3/health_assessment"
API_KEY = os.getenv("API_KEY")

with open("img.jpeg", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

payload = json.dumps({
  "images": [
    encoded_string
  ],
  "latitude": 49.207,
  "longitude": 16.608,
  "similar_images": True
})
headers = {
  'Api-Key': API_KEY,
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)

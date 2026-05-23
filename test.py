import requests

url = "http://127.0.0.1:8080/predict"
data = {
    "home_team": "Japan",
    "away_team": "South Korea",
    "tournament": "FIFA World Cup qualification"
}

response = requests.post(url, json=data)
print(response.json())

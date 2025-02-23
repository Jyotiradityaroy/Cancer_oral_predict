import requests

url = "http://127.0.0.1:5000/predict"
files = {"file": open("test_image.jpg", "rb")}  # Replace with your image file
response = requests.post(url, files=files)

print(response.json())  # Output: {"prediction": "Cancerous" or "Non-Cancerous"}

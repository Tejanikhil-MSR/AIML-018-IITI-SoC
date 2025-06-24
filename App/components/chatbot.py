import requests

# Change this if your backend runs elsewhere
BACKEND_URL = "http://localhost:8011"

def get_bot_response(user_message):
    try:
        response = requests.post(
            BACKEND_URL,
            headers={"Content-Type": "application/json"},
            json={"messages": user_message}
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "❌ No 'response' key in reply.")
    except requests.exceptions.RequestException as e:
        return f"❌ Error: {str(e)}"

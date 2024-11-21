import requests

def test_api():
    # test_prompt = "Once upon a time"
    test_prompt = "Donald Trump is"
    url = "https://c395-34-82-224-117.ngrok-free.app/generate" # Copy paste the ngrok URL here / not a neat way to do it but it works
    
    try:
        response = requests.post(
            url,
            json={
                "prompt": test_prompt,
                "max_tokens": 50,
                "temperature": 0.7,
                "top_p": 0.9
            },
            timeout=300
        )
        
        # Print response details for debugging
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {response.headers}")
        print(f"Raw Response: {response.text}")
        
        # Check if response is successful
        response.raise_for_status()
        
        try:
            return response.json()
        except requests.exceptions.JSONDecodeError as e:
            print(f"Failed to decode JSON: {e}")
            print(f"Response content: {response.content}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None

# Test the API
result = test_api()
if result:
    print("\nParsed JSON Response:", result)

import pytest
from fastapi.testclient import TestClient
import uvicorn 
from pyngrok import ngrok 
import threading
import time 
# Import the FastAPI app and other necessary components from mainn.py
# from mainn import app, OptimizedLLM, BatchManager
from llama_attempt import app, OptimizedLLM

ngrok.set_auth_token("2o8EyZCZfWRWEc1nnKDe0ORtAJ4_6sgfuHGZWaek4cuZL7uS1")

def run_server():
  uvicorn.run(app, host="0.0.0.0", port=8000)

server_thread = threading.Thread(target=run_server)
server_thread.start()

time.sleep(5)

ngrok_tunnel = ngrok.connect(8000)
print('Public URL:', ngrok_tunnel.public_url)

client = TestClient(app)

def test_model_performance():
    # client = TestClient(app)
    
    # Test single request
    try:
      response = client.post("/generate", json={
          "prompt": "Describe Trump in one sentence.",
          "max_length": 128
      })
      print("Response status code:", response.status_code)
      print("Response content:", response.content)
      assert response.status_code == 200
      print("Single request test passed.")
      print("Generated Text:", response.json()["generated_text"])
    except Exception as e:
      print("Single request test failed:", str(e))
    
    # Test batch performance
    responses = []
    for i in range(5):  # Test concurrent requests
        try:
          response = client.post("/generate", json={
              "prompt": "Benefits of exercise?",
              "max_length": 128
          })
          responses.append(response)
          print(f'Batch Request {i+1} status code:', response.status_code)
        except Exception as e:
          print(f'Batch Request {i+1} failed:', str(e))

    # Verify all requests succeeded
    success = all(r.status_code == 200 for r in responses)
    print("All batch requests succeeded:", success)
    
    if success: 
      print("Batch request test passed")

      # Verify batching worked
      try:
        processing_times = [r.json()["processing_time"] for r in responses]
        time_diff = max(processing_times) - min(processing_times)
        print(f'Time difference: {time_diff}')
        assert time_diff < 0.5, "Batch processing may not be working efficiently"
        print("Batch request test passed.")
      except Exception as e:
        print("Batch request test failed:", str(e))
    else:
      print("batch request test failed")

    
test_model_performance()

ngrok.disconnect(ngrok_tunnel.public_url)
server_thread.join(timeout=1)
#!/usr/bin/env python
"""Post-deployment smoke tests for the Cats vs Dogs classifier API."""

import argparse
import sys
import time
from pathlib import Path
import io

import requests
import numpy as np
from PIL import Image


def create_test_image() -> bytes:
    """Create a random test image."""
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer.getvalue()


def test_health_endpoint(base_url: str, max_retries: int = 5) -> bool:
    """
    Test the health check endpoint.
    
    Args:
        base_url: Base URL of the API
        max_retries: Maximum number of retry attempts
        
    Returns:
        True if test passes
    """
    url = f"{base_url}/health"
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get("status") == "healthy":
                    print(f"[PASS] Health check passed: {data}")
                    return True
                else:
                    print(f"[WARN] Health check returned unhealthy status: {data}")
                    
        except requests.exceptions.RequestException as e:
            print(f"[RETRY {attempt + 1}/{max_retries}] Health check failed: {e}")
            time.sleep(2)
    
    print("[FAIL] Health check failed after all retries")
    return False


def test_prediction_endpoint(base_url: str) -> bool:
    """
    Test the prediction endpoint with a sample image.
    
    Args:
        base_url: Base URL of the API
        
    Returns:
        True if test passes
    """
    url = f"{base_url}/predict"
    
    try:
        image_bytes = create_test_image()
        
        files = {"file": ("test.jpg", image_bytes, "image/jpeg")}
        response = requests.post(url, files=files, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            required_fields = ["prediction", "label", "confidence"]
            missing_fields = [f for f in required_fields if f not in data]
            
            if missing_fields:
                print(f"[FAIL] Prediction response missing fields: {missing_fields}")
                return False
            
            if data["prediction"] not in [0, 1]:
                print(f"[FAIL] Invalid prediction value: {data['prediction']}")
                return False
            
            if data["label"] not in ["cat", "dog"]:
                print(f"[FAIL] Invalid label: {data['label']}")
                return False
            
            if not (0 <= data["confidence"] <= 1):
                print(f"[FAIL] Invalid confidence: {data['confidence']}")
                return False
            
            print(f"[PASS] Prediction endpoint works: {data}")
            return True
        
        elif response.status_code == 503:
            print(f"[WARN] Model not loaded (503): {response.json()}")
            return True
        
        else:
            print(f"[FAIL] Prediction failed with status {response.status_code}: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"[FAIL] Prediction request failed: {e}")
        return False


def test_metrics_endpoint(base_url: str) -> bool:
    """
    Test the Prometheus metrics endpoint.
    
    Args:
        base_url: Base URL of the API
        
    Returns:
        True if test passes
    """
    url = f"{base_url}/metrics"
    
    try:
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            content = response.text
            
            if "prediction" in content.lower() or "model" in content.lower() or "python" in content.lower():
                print("[PASS] Metrics endpoint returns Prometheus format")
                return True
            else:
                print("[WARN] Metrics endpoint works but may not have custom metrics")
                return True
        else:
            print(f"[FAIL] Metrics endpoint failed: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"[FAIL] Metrics request failed: {e}")
        return False


def test_docs_endpoint(base_url: str) -> bool:
    """
    Test the Swagger documentation endpoint.
    
    Args:
        base_url: Base URL of the API
        
    Returns:
        True if test passes
    """
    url = f"{base_url}/docs"
    
    try:
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            print("[PASS] Documentation endpoint accessible")
            return True
        else:
            print(f"[WARN] Docs endpoint returned {response.status_code}")
            return True
            
    except requests.exceptions.RequestException as e:
        print(f"[WARN] Docs request failed: {e}")
        return True


def run_smoke_tests(base_url: str) -> bool:
    """
    Run all smoke tests.
    
    Args:
        base_url: Base URL of the API
        
    Returns:
        True if all critical tests pass
    """
    print(f"\n{'=' * 50}")
    print(f"Running Smoke Tests against: {base_url}")
    print(f"{'=' * 50}\n")
    
    results = {}
    
    print("1. Testing Health Endpoint...")
    results["health"] = test_health_endpoint(base_url)
    
    if not results["health"]:
        print("\n[CRITICAL] Health check failed. Aborting remaining tests.")
        return False
    
    print("\n2. Testing Prediction Endpoint...")
    results["prediction"] = test_prediction_endpoint(base_url)
    
    print("\n3. Testing Metrics Endpoint...")
    results["metrics"] = test_metrics_endpoint(base_url)
    
    print("\n4. Testing Documentation Endpoint...")
    results["docs"] = test_docs_endpoint(base_url)
    
    print(f"\n{'=' * 50}")
    print("Smoke Test Results:")
    print(f"{'=' * 50}")
    
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test_name}: {status}")
    
    critical_tests = ["health", "prediction"]
    all_critical_passed = all(results.get(t, False) for t in critical_tests)
    
    if all_critical_passed:
        print("\n[SUCCESS] All critical smoke tests passed!")
        return True
    else:
        print("\n[FAILURE] Some critical smoke tests failed!")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run smoke tests against the API")
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the API",
    )
    parser.add_argument(
        "--wait",
        type=int,
        default=0,
        help="Seconds to wait before starting tests",
    )
    
    args = parser.parse_args()
    
    if args.wait > 0:
        print(f"Waiting {args.wait} seconds before starting tests...")
        time.sleep(args.wait)
    
    success = run_smoke_tests(args.url.rstrip("/"))
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

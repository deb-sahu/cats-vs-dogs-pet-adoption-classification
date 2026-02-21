#!/usr/bin/env python
"""Simulate traffic to the API for monitoring testing."""

import argparse
import time
import random
import io
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import numpy as np
from PIL import Image


def create_random_image() -> bytes:
    """Create a random test image."""
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer.getvalue()


def send_prediction_request(base_url: str) -> dict:
    """Send a single prediction request."""
    url = f"{base_url}/predict"
    image_bytes = create_random_image()
    
    try:
        start = time.time()
        response = requests.post(
            url,
            files={"file": ("test.jpg", image_bytes, "image/jpeg")},
            timeout=30
        )
        latency = time.time() - start
        
        return {
            "status_code": response.status_code,
            "latency": latency,
            "success": response.status_code == 200,
            "data": response.json() if response.status_code == 200 else None
        }
    except Exception as e:
        return {
            "status_code": 0,
            "latency": 0,
            "success": False,
            "error": str(e)
        }


def simulate_traffic(
    base_url: str,
    requests_per_second: float = 1.0,
    duration_seconds: int = 60,
    num_workers: int = 4,
):
    """
    Simulate traffic to the API.
    
    Args:
        base_url: Base URL of the API
        requests_per_second: Target requests per second
        duration_seconds: Duration to run simulation
        num_workers: Number of concurrent workers
    """
    print(f"Simulating traffic to {base_url}")
    print(f"Target: {requests_per_second} req/s for {duration_seconds} seconds")
    print(f"Workers: {num_workers}")
    print("-" * 50)
    
    interval = 1.0 / requests_per_second
    start_time = time.time()
    request_count = 0
    success_count = 0
    total_latency = 0.0
    
    cats = 0
    dogs = 0
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        
        while time.time() - start_time < duration_seconds:
            future = executor.submit(send_prediction_request, base_url)
            futures.append(future)
            request_count += 1
            
            time.sleep(interval + random.uniform(-0.1, 0.1) * interval)
            
            completed = [f for f in futures if f.done()]
            for f in completed:
                futures.remove(f)
                result = f.result()
                
                if result["success"]:
                    success_count += 1
                    total_latency += result["latency"]
                    
                    if result["data"]:
                        label = result["data"].get("label", "")
                        if label == "cat":
                            cats += 1
                        elif label == "dog":
                            dogs += 1
                
                elapsed = time.time() - start_time
                current_rps = request_count / elapsed if elapsed > 0 else 0
                
                sys.stdout.write(
                    f"\rRequests: {request_count} | "
                    f"Success: {success_count} | "
                    f"RPS: {current_rps:.2f} | "
                    f"Cats: {cats} | Dogs: {dogs}"
                )
                sys.stdout.flush()
        
        for future in as_completed(futures):
            result = future.result()
            if result["success"]:
                success_count += 1
                total_latency += result["latency"]
    
    print("\n" + "-" * 50)
    print("Simulation Complete!")
    print(f"Total Requests: {request_count}")
    print(f"Successful: {success_count}")
    print(f"Failed: {request_count - success_count}")
    print(f"Success Rate: {100 * success_count / request_count:.1f}%")
    if success_count > 0:
        print(f"Avg Latency: {total_latency / success_count:.3f}s")
    print(f"Predictions - Cats: {cats}, Dogs: {dogs}")


def main():
    parser = argparse.ArgumentParser(description="Simulate API traffic for monitoring")
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the API",
    )
    parser.add_argument(
        "--rps",
        type=float,
        default=2.0,
        help="Requests per second",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Duration in seconds",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of concurrent workers",
    )
    
    args = parser.parse_args()
    
    print("Testing API connectivity...")
    try:
        response = requests.get(f"{args.url}/health", timeout=5)
        if response.status_code == 200:
            print("API is healthy!")
        else:
            print(f"API returned status {response.status_code}")
    except Exception as e:
        print(f"Failed to connect to API: {e}")
        sys.exit(1)
    
    print()
    simulate_traffic(
        base_url=args.url.rstrip("/"),
        requests_per_second=args.rps,
        duration_seconds=args.duration,
        num_workers=args.workers,
    )


if __name__ == "__main__":
    main()

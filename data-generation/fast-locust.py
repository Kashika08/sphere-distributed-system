"""
 Microservices Demo Load Test
 - Description: Simulates real-world traffic patterns for a microservices-based e-commerce application
 - Target: Frontend at http://192.168.49.2:30001
 - Loop Interval: 8 minutes per cycle (then repeat)
 - Goal: Spike load to trigger 95th‑percentile violations for K8s HPA (80% upscale, 20% downscale)
"""

import time
import random
import json
from locust import HttpUser, task, between, LoadTestShape, events
from datetime import datetime
import gevent

# --- Global counters for request statistics ---
TOTAL_REQUESTS = 0
SLOW_COUNT     = 0  # responses > 2000ms

# --- Average Response Time Logger ---
def setup_response_time_logger(environment):
    """Background task to log average response times periodically"""
    def logger_task():
        with open('response_times.log', 'w') as f:
            f.write(f"Load Test Started: {datetime.now()}\n\n")
        while environment.runner.state != "stopped":
            gevent.sleep(10)
            stats = environment.runner.stats.total
            if stats.num_requests > 0:
                avg_time = round(stats.avg_response_time, 2)
                try:
                    pct_95 = round(stats.get_response_time_percentile(0.95), 2)
                except AttributeError:
                    pct_95 = round(stats.get_percentile(95.0), 2)
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                log_entry = (
                    f"{timestamp} - Avg Response: {avg_time}ms | "
                    f"95th Percentile: {pct_95}ms | Users: {environment.runner.user_count}\n"
                )
                with open('response_times.log', 'a') as f:
                    f.write(log_entry)
    gevent.spawn(logger_task)

# --- Custom Load Shape with 8‑minute Cycle ---
class CustomLoadShape(LoadTestShape):
    stages = [
        {"duration": 60, "users": 200,  "spawn_rate": 2},
        {"duration": 60, "users": 600,  "spawn_rate": 5},
        {"duration": 60, "users": 1000, "spawn_rate": 10},
        {"duration": 60, "users": 1500, "spawn_rate": 15},
        {"duration": 60, "users": 2000, "spawn_rate": 20},
        {"duration": 60, "users": 1500, "spawn_rate": 15},
        {"duration": 60, "users": 1000, "spawn_rate": 10},
        {"duration": 60, "users": 500,  "spawn_rate": 5},
    ]
    total_duration = sum(stage["duration"] for stage in stages)

    def tick(self):
        elapsed = self.get_run_time() % self.total_duration
        for stage in self.stages:
            if elapsed < stage["duration"]:
                return stage["users"], stage["spawn_rate"]
            elapsed -= stage["duration"]
        return self.stages[-1]["users"], self.stages[-1]["spawn_rate"]

# --- Product categories for browsing simulation ---
categories = ["formal", "smelly", "large", "short", "toes", "magic", "blue", "brown", "green"]

class WebsiteUser(HttpUser):
    wait_time = between(1, 5)

    def on_start(self):
        self.client.keep_alive = False
        self.client.headers.update({"Connection": "close"})

    @task(25)
    def browse_homepage(self):
        with self.client.get("/", catch_response=True, headers={"Connection": "close"}) as response:
            if response.status_code != 200:
                response.failure(f"Homepage failed with status code: {response.status_code}")

    @task(13)
    def browse_category(self):
        category = random.choice(categories)
        with self.client.get(f"/category.html?tags={category}", catch_response=True, headers={"Connection": "close"}) as response:
            if response.status_code != 200:
                response.failure(f"Category browsing failed with status code: {response.status_code}")

# --- Event hooks ---
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    print("Starting load test for microservices demo application")
    print("Cycle length: 8 minutes, then repeat")
    setup_response_time_logger(environment)

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    print("\nLoad test stopped by user interruption")
    # Print final summary
    print(f"Total requests: {TOTAL_REQUESTS}")
    print(f"Slow responses (>2000ms): {SLOW_COUNT}")
    if TOTAL_REQUESTS > 0:
        pct = round((SLOW_COUNT / TOTAL_REQUESTS) * 100, 2)
        print(f"Percentage slow: {pct}%")

@events.request.add_listener
def on_request(request_type, name, response_time, response_length, response, context, exception, **kwargs):
    global TOTAL_REQUESTS, SLOW_COUNT
    TOTAL_REQUESTS += 1

    if exception:
        print(f"Request to {name} failed with exception: {exception}")
    elif response.status_code >= 400:
        print(f"Request to {name} failed with status code: {response.status_code}")
    # Modified threshold to >2000ms
    elif response_time > 2000:
        SLOW_COUNT += 1
        print(f"SLOW REQUEST: {name} took {response_time}ms to complete")

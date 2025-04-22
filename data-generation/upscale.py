"""
 Microservices Demo Load Test
 - Description: Simulates real-world traffic patterns for a microservices-based e-commerce application
 - Target: Frontend at http://192.168.49.2:30001
"""

import time
import random
import json
from locust import HttpUser, task, between, LoadTestShape, events
from datetime import datetime
import gevent

TOTAL_REQUESTS = 0
SLOW_COUNT = 0

# --- Average Response Time Logger ---
def setup_response_time_logger(environment):
    """Background task to log average response times periodically"""
    def logger_task():
        with open('response_times.log', 'w') as f:  # Create/clear file at start
            f.write(f"Load Test Started: {datetime.now()}\n\n")
        while environment.runner.state != "stopped":
            gevent.sleep(10)  # Log every 10 seconds
            stats = environment.runner.stats.total
            if stats.num_requests > 0:
                avg_time = round(stats.avg_response_time, 2)
                try:
                    pct_95 = round(stats.get_response_time_percentile(0.95), 2)
                except AttributeError:
                    # Fallback if API differs
                    pct_95 = round(stats.get_percentile(95.0), 2)
                #pct_95 = round(stats.get_percentile(95.0), 2)
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                log_entry = (
                    f"{timestamp} - Avg Response: {avg_time}ms | "
                    f"95th Percentile: {pct_95}ms | Users: {environment.runner.user_count}\n"
                )
                with open('response_times.log', 'a') as f:
                    f.write(log_entry)
    gevent.spawn(logger_task)

# --- Custom Load Shape with Infinite Cycling ---
class CustomLoadShape(LoadTestShape):
    stages = [
        {"duration": 150, "users": 200, "spawn_rate": 1.33},
        {"duration": 150, "users": 350, "spawn_rate": 1.0},
        {"duration": 500, "users": 600, "spawn_rate": 1.5},
    ]
    total_duration = sum(stage['duration'] for stage in stages)

    def tick(self):
        """Determines user count and spawn rate at each time interval, cycling indefinitely"""
        elapsed = self.get_run_time()
        # Loop elapsed time within one full cycle
        elapsed = elapsed % self.total_duration
        for stage in self.stages:
            if elapsed < stage["duration"]:
                return stage["users"], stage["spawn_rate"]
            elapsed -= stage["duration"]
        # Fallback (should not reach here)
        return self.stages[-1]["users"], self.stages[-1]["spawn_rate"]

# --- Product categories for browsing simulation ---
categories = ["formal", "smelly", "large", "short", "toes", "magic", "blue", "brown", "green"]

class WebsiteUser(HttpUser):
    wait_time = between(1, 5)  # Realistic wait time between actions

    def on_start(self):
        """Initialize user session with login"""
        self.client.keep_alive = False
        self.client.headers.update({"Connection": "close"})

    @task(25)
    def browse_homepage(self):
        """Browse the homepage and featured products"""
        with self.client.get("/", catch_response=True, headers={"Connection": "close"}) as response:
            if response.status_code != 200:
                response.failure(f"Homepage failed with status code: {response.status_code}")

    @task(13)
    def browse_category(self):
        """Browse a random product category"""
        category = random.choice(categories)
        with self.client.get(f"/category.html?tags={category}", catch_response=True, headers={"Connection": "close"}) as response:
            if response.status_code != 200:
                response.failure(f"Category browsing failed with status code: {response.status_code}")

# --- Event hooks for logging and monitoring ---
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    print("Starting load test for microservices demo application")
    print(f"Target maximum users: 1100")
    print(f"Test will run indefinitely cycling through load stages")
    setup_response_time_logger(environment)

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    print("Load test stopped by user interruption")
    print(f"Total requests: {TOTAL_REQUESTS}")
    print(f"Slow responses (>500ms): {SLOW_COUNT}")
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
    elif response_time > 500:
        SLOW_COUNT += 1
        print(f"SLOW REQUEST: {name} took {response_time}ms to complete")

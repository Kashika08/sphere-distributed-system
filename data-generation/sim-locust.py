"""
Microservices Demo Load Test
- Description: Simulates real-world traffic patterns for a microservices-based e-commerce application
- Target: Frontend at http://192.168.49.2:30001
- Maximum Users: 700
"""

import time
import random
import json
from locust import HttpUser, task, between, LoadTestShape, events

# Custom load shape to simulate real-world traffic patterns
class CustomLoadShape(LoadTestShape):
    stages = [
        {"duration": 250, "users": 125, "spawn_rate": 0.5},     # Initial warm-up phase
        {"duration": 250, "users": 250, "spawn_rate": 0.5},   # Morning traffic ramp-up
        {"duration": 291, "users": 425, "spawn_rate": 0.6},   # Mid-day peak
        {"duration": 250, "users": 275, "spawn_rate": 0.6},    # Post-lunch dip
        {"duration": 300, "users": 500, "spawn_rate": 0.75},   # Afternoon traffic surge
        {"duration": 563, "users": 700, "spawn_rate": 0.55},   # Evening peak (maximum load)
        {"duration": 350, "users": 350, "spawn_rate": 1.0},   # Early evening decline
        {"duration": 250, "users": 250, "spawn_rate": 0.4},    # Night traffic
        {"duration": 250, "users": 100, "spawn_rate": 0.6},     # Cool down period
    ]
    
    def tick(self):
        """Determines user count and spawn rate at each time interval"""
        elapsed = self.get_run_time()
        
        for stage in self.stages:
            if elapsed < stage["duration"]:
                target_users = stage["users"]
                spawn_rate = stage["spawn_rate"]
                return target_users, spawn_rate
            elapsed -= stage["duration"]
        
        return 0, 0  # End the test when all stages are complete


# Product categories for browsing simulation
categories = ["formal", "smelly", "large", "short","toes", "magic", "blue", "brown", "green"]

class WebsiteUser(HttpUser):
    wait_time = between(1, 5)  # Realistic wait time between actions
    
    def on_start(self):
        """Initialize user session with login"""
        self.client.keep_alive = False
        self.client.headers.update({"Connection": "close"})
        pass
    
    
    @task(25)
    def browse_homepage(self):
        """Browse the homepage and featured products"""
        headers = {'Connection': 'close'}
        with self.client.get("/", catch_response=True, headers=headers) as response:
            if response.status_code != 200:
                response.failure(f"Homepage failed with status code: {response.status_code}")
    
    @task(13)
    def browse_category(self):
        """Browse a random product category"""
        category = random.choice(categories)
        headers = {'Connection': 'close'}
        with self.client.get(f"/category.html?tags={category}", catch_response=True, headers=headers) as response:
            if response.status_code != 200:
                response.failure(f"Category browsing failed with status code: {response.status_code}")


# Event hooks for additional logging and monitoring
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    print("Starting load test for microservices demo application")
    print(f"Target maximum users: 700")
    print(f"Test will simulate a full day traffic pattern")

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    print("Load test completed")
    print(f"Total test duration: {sum(stage['duration'] for stage in CustomLoadShape.stages)} seconds")

# Custom request handler to log performance metrics
@events.request.add_listener
def on_request(request_type, name, response_time, response_length, response, context, exception, **kwargs):
    if exception:
        print(f"Request to {name} failed with exception: {exception}")
    elif response.status_code >= 400:
        print(f"Request to {name} failed with status code: {response.status_code}")
    elif response_time > 1000:  # Log slow responses (>1000ms)
        print(f"SLOW REQUEST: {name} took {response_time}ms to complete")

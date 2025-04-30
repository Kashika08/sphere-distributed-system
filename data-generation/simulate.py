"""
 Microservices Demo Load Test
 - Description: Simulates real-world traffic patterns for a microservices-based e-commerce application
 - Target: Frontend at http://192.168.49.2:30001
 - Maximum Users: 400
"""

import random
from locust import HttpUser, task, between, LoadTestShape, events
from datetime import datetime
import gevent

TOTAL_REQUESTS = 0
SLOW_COUNT = 0

# Globals for per-second aggregation
_current_ts = None
_buffer = []  # tuples of (avg_response, pct95, this_response)

ENV = None  # hold Locust Environment so we can read global stats

def flush_buffer():
    """Write one aggregated line for the last second, then clear."""
    global _current_ts, _buffer
    if not _current_ts or not _buffer:
        _current_ts = None
        _buffer = []
        return

    count = len(_buffer)
    avg_of_avgs = round(sum(x[0] for x in _buffer) / count, 2)
    avg_of_95s  = round(sum(x[1] for x in _buffer) / count, 2)
    avg_resp    = round(sum(x[2] for x in _buffer) / count, 2)

    with open('response_times.log', 'a') as f:
        f.write(f"{_current_ts},{avg_of_avgs},{avg_of_95s},{avg_resp}\n")

    _current_ts = None
    _buffer = []

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    global ENV
    ENV = environment
    # initialize/overwrite log
    with open('response_times.log', 'w') as f:
        f.write("timestamp,avg_response_time_ms,95th_percentile_ms,response_time_ms\n")

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    # flush whatever is left
    flush_buffer()
    print(f"Load test stopped. Total requests: {TOTAL_REQUESTS}, slow: {SLOW_COUNT}")

@events.request.add_listener
def on_request(request_type, name, response_time, response_length, response, context, exception, **kwargs):
    global TOTAL_REQUESTS, SLOW_COUNT, _current_ts, _buffer

    TOTAL_REQUESTS += 1
    if not exception and response and response.status_code < 400 and response_time > 500:
        SLOW_COUNT += 1

    # read the rolling globals
    stats = ENV.runner.stats.total
    avg_time = round(stats.avg_response_time, 2)
    try:
        pct_95 = round(stats.get_response_time_percentile(0.95), 2)
    except AttributeError:
        pct_95 = round(stats.get_percentile(95.0), 2)

    # group by second
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if _current_ts is None:
        _current_ts = ts
    elif ts != _current_ts:
        # time moved to next second → flush previous
        flush_buffer()
        _current_ts = ts

    # collect this request's data
    _buffer.append((avg_time, pct_95, response_time))

# --- Custom Load Shape with Infinite Cycling ---
class CustomLoadShape(LoadTestShape):
    stages = [
        {"duration":150, "users":200,  "spawn_rate":1.33},
        {"duration":150, "users":350,  "spawn_rate":1.0},
        {"duration":100, "users":500,  "spawn_rate":1.5},
        {"duration":200, "users":325,  "spawn_rate":0.875},
        {"duration":100, "users":800,  "spawn_rate":4.75},
        {"duration":62.5,"users":900,  "spawn_rate":1.6},
        {"duration":130,"users":1100, "spawn_rate":2.0},
        {"duration":150,"users":800,  "spawn_rate":2.0},
        {"duration":100,"users":1200, "spawn_rate":4.0},
        {"duration":250,"users":700,  "spawn_rate":2.0},
        {"duration":300,"users":400,  "spawn_rate":1.0},
    ]
    total_duration = sum(s["duration"] for s in stages)

    def tick(self):
        elapsed = self.get_run_time() % self.total_duration
        for s in self.stages:
            if elapsed < s["duration"]:
                return s["users"], s["spawn_rate"]
            elapsed -= s["duration"]
        return self.stages[-1]["users"], self.stages[-1]["spawn_rate"]

# --- Simulated User Behavior ---
categories = ["formal","smelly","large","short","toes","magic","blue","brown","green"]

class WebsiteUser(HttpUser):
    wait_time = between(1, 5)

    def on_start(self):
        self.client.keep_alive = False
        self.client.headers.update({"Connection":"close"})

    @task(25)
    def browse_homepage(self):
        with self.client.get("/", catch_response=True) as resp:
            if resp.status_code != 200:
                resp.failure(f"Bad home: {resp.status_code}")

    @task(13)
    def browse_category(self):
        cat = random.choice(categories)
        with self.client.get(f"/category.html?tags={cat}", catch_response=True) as resp:
            if resp.status_code != 200:
                resp.failure(f"Bad cat: {resp.status_code}")


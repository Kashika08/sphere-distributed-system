import time
import random
import json
from locust import HttpUser, task, between, LoadTestShape, events
from datetime import datetime
import gevent

TOTAL_REQUESTS = 0
SLOW_COUNT = 0
ENV = None  # Locust Environment

# --- In-memory per-second aggregates ---
per_second_data = {}  # { 'YYYY-MM-DD HH:MM:SS': {'sum_avg':..., 'sum_95':..., 'sum_resp':..., 'count':...} }
last_flush_second = None

# --- Load NASA 10s‐interval counts ---
with open("nasa_10s_counts.txt", "r") as f:
    NASA_COUNTS = [int(line.strip()) for line in f if line.strip()]

# --- Average and flush logger ---
def setup_aggregator(environment):
    """Background task: every second flush previous-second aggregates to file"""  
    # create/clear file and write header
    with open('response_times.log', 'w') as f:
        f.write("timestamp,avg_response_time_ms,95th_percentile_ms,avg_response_time_this_sec_ms,request_count\n")

    def aggregator_task():
        global last_flush_second
        while environment.runner.state != 'stopped':
            gevent.sleep(1)
            now_sec = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            if last_flush_second is None:
                last_flush_second = now_sec
                continue
            if now_sec != last_flush_second:
                data = per_second_data.pop(last_flush_second, None)
                if data:
                    avg_avg = round(data['sum_avg'] / data['count'], 2)
                    avg_95 = round(data['sum_95'] / data['count'], 2)
                    avg_resp = round(data['sum_resp'] / data['count'], 2)
                    with open('response_times.log', 'a') as f:
                        f.write(f"{last_flush_second},{avg_avg},{avg_95},{avg_resp},{data['count']}\n")
                last_flush_second = now_sec
        # flush remaining
        data = per_second_data.pop(last_flush_second, None)
        if data:
            avg_avg = round(data['sum_avg'] / data['count'], 2)
            avg_95 = round(data['sum_95'] / data['count'], 2)
            avg_resp = round(data['sum_resp'] / data['count'], 2)
            with open('response_times.log', 'a') as f:
                f.write(f"{last_flush_second},{avg_avg},{avg_95},{avg_resp},{data['count']}\n")

    gevent.spawn(aggregator_task)

# --- Custom Load Shape Driven by NASA Trace ---
class NASA10sLoadShape(LoadTestShape):
    sample_interval = 10    # seconds
    scale_factor    = 50.0  # magnify load
    total_steps     = len(NASA_COUNTS)

    def tick(self):
        elapsed = int(self.get_run_time())
        step = elapsed // self.sample_interval
        if step < self.total_steps:
            count = NASA_COUNTS[step]
            users = int(count * self.scale_factor)
            spawn_rate = max(1, users // 10)
            return users, spawn_rate
        return None

# --- Product categories ---
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
                response.failure(f"Homepage failed: {response.status_code}")

    @task(13)
    def browse_category(self):
        category = random.choice(categories)
        with self.client.get(f"/category.html?tags={category}", catch_response=True, headers={"Connection": "close"}) as response:
            if response.status_code != 200:
                response.failure(f"Category {category} failed: {response.status_code}")

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    global ENV
    ENV = environment
    print("Starting NASA-driven load test with per-second logging")
    setup_aggregator(environment)

@events.request.add_listener
def on_request(request_type, name, response_time, response_length, response, context, exception, **kwargs):
    global TOTAL_REQUESTS, SLOW_COUNT
    TOTAL_REQUESTS += 1
    if not exception and response and response.status_code < 400 and response_time > 500:
        SLOW_COUNT += 1

    # fetch current global stats
    if ENV and ENV.runner:
        stats = ENV.runner.stats.total
        avg_time = round(stats.avg_response_time, 2)
        try:
            pct_95 = round(stats.get_response_time_percentile(0.95), 2)
        except AttributeError:
            pct_95 = round(stats.get_percentile(95.0), 2)
    else:
        avg_time = pct_95 = 0.0

    # accumulate per-second
    now_sec = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    entry = per_second_data.setdefault(now_sec, { 'sum_avg':0, 'sum_95':0, 'sum_resp':0, 'count':0 })
    entry['sum_avg'] += avg_time
    entry['sum_95'] += pct_95
    entry['sum_resp'] += response_time
    entry['count'] += 1

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    # remaining flush happens in aggregator
    print("Load test stopped")
    print(f"Total requests: {TOTAL_REQUESTS}")
    print(f"Slow responses (>500ms): {SLOW_COUNT}")
    if TOTAL_REQUESTS:
        pct = round((SLOW_COUNT / TOTAL_REQUESTS) * 100, 2)
        print(f"Percent slow: {pct}%")


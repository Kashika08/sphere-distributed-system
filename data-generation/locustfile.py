"""
UI‑driven Load Test for Sock‑Shop Frontend
- Launch via: locust -f locustfile.py --host http://<FRONTEND_NODE_IP>:30001
- Then open http://localhost:8089 to adjust User count & Spawn rate
- Smoothly paces requests to a global max of 200 RPS (no bursts)
"""

import random
from locust import HttpUser, task, events
from datetime import datetime
import gevent

# Maximum global requests per second
MAX_RPS = 200

# --- Average Response Time Logger ---
def setup_response_time_logger(environment):
    def logger_task():
        with open("response_times.log", "w") as f:
            f.write(f"Load Test Started: {datetime.now()}\n\n")
        while environment.runner.state != "stopped":
            gevent.sleep(10)
            stats = environment.runner.stats.total
            if stats.num_requests > 0:
                avg_time = round(stats.avg_response_time, 2)
                try:
                    pct_dict = stats.get_current_response_time_percentiles()
                    pct_95 = round(pct_dict.get("95", 0), 2)
                except AttributeError:
                    pct_95 = "N/A"
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                entry = (
                    f"{timestamp} - Avg: {avg_time}ms | "
                    f"95th: {pct_95}ms | Users: {environment.runner.user_count}\n"
                )
                with open("response_times.log", "a") as f:
                    f.write(entry)
    gevent.spawn(logger_task)

# --- Product categories for browsing simulation ---
categories = ["formal", "casual", "sports", "summer", "winter", "blue", "green", "red"]

class WebsiteUser(HttpUser):
    # override wait() to pace users so total RPS = MAX_RPS
    def wait(self):
        try:
            user_count = self.environment.runner.user_count
        except Exception:
            user_count = 1
        # per-user wait time in seconds
        wait_time = user_count / MAX_RPS
        gevent.sleep(wait_time)

    def on_start(self):
        self.client.keep_alive = False
        self.client.headers.update({"Connection": "close"})

    @task(50)
    def browse_home(self):
        with self.client.get("/", catch_response=True) as r:
            if r.status_code != 200:
                r.failure(f"Homepage failed: {r.status_code}")

    @task(30)
    def browse_category(self):
        cat = random.choice(categories)
        with self.client.get(f"/category.html?tags={cat}", catch_response=True) as r:
            if r.status_code != 200:
                r.failure(f"Category {cat} failed: {r.status_code}")

# --- Lifecycle Hooks ---
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    print(f"Starting load test — capping global RPS at {MAX_RPS}")
    setup_response_time_logger(environment)

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    print("Load test stopped")

@events.request.add_listener
def on_request(**kwargs):
    name = kwargs.get("name")
    response_time = kwargs.get("response_time")
    exception = kwargs.get("exception")
    response = kwargs.get("response")
    if exception:
        print(f"EXCEPT: {name} → {exception}")
    elif response is not None and getattr(response, "status_code", 0) >= 400:
        print(f"FAIL: {name} → {response.status_code}")
    elif response_time is not None and response_time > 1000:
        print(f"SLOW: {name} took {response_time}ms")

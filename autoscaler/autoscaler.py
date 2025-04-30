import redis
import subprocess
import time
import json
import math

# --- Configuration ---
HIGH_THRESHOLD = 80.0
LOW_THRESHOLD = 20.0
CPU_BUFFER = 10.0

MAX_REPLICAS = 4
MIN_REPLICAS = 1
DOWNSCALE_COOLDOWN_SECONDS = 60  # Only for downscale

REDIS_HOST = 'localhost'
REDIS_PORT = 6379
NAMESPACE = 'sock-shop'
DEPLOYMENT = 'front-end'

DOWNSCALE_COOLDOWN_KEY = f'downscale_triggered:{DEPLOYMENT}'

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"

rdb = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# --- Utility Functions ---
def extract_predictions(entry: str):
    try:
        if "Predictions:" not in entry:
            return []
        prediction_str = entry.split("Predictions:")[1].strip()
        return list(map(float, prediction_str.split(",")))
    except Exception as e:
        print(f"[ERROR] Failed to parse predictions: {e}")
        return []

def get_current_replicas():
    try:
        result = subprocess.run(
            ["kubectl", "get", "deployment", DEPLOYMENT, "-n", NAMESPACE, "-o", "json"],
            capture_output=True, text=True, check=True
        )
        deployment_info = json.loads(result.stdout)
        replicas = deployment_info["status"].get("replicas", MIN_REPLICAS)
        return int(replicas)
    except Exception as e:
        print(f"[ERROR] Fetching current replicas from Kubernetes: {e}")
        return MIN_REPLICAS

def scale_replicas(target):
    print(f"🔁 Scaling {DEPLOYMENT} to {target} replicas...")
    try:
        subprocess.run(
            ["kubectl", "scale", f"deployment/{DEPLOYMENT}", "-n", NAMESPACE, f"--replicas={target}"],
            check=True
        )
        print("✅ Scale operation triggered successfully.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to scale deployment: {e.stderr or e}")

def loud_log(message, color=GREEN):
    print(f"{'='*50}")
    print(f"{BOLD}{color}{message}{RESET}")
    print(f"{'='*50}\n")

def is_downscale_in_cooldown():
    return rdb.exists(DOWNSCALE_COOLDOWN_KEY)

# --- Main Scaling Logic ---
def process_prediction(key):
    entry = rdb.get(key)
    if not entry:
        return

    predictions = extract_predictions(entry)
    if len(predictions) < 3:
        print(f"[WARN] Not enough predictions in {key}")
        return

    avg_last_3 = sum(predictions[-3:]) / 3
    current_replicas = get_current_replicas()

    if current_replicas == 0:
        print("⚠️ Replica count is zero, skipping scaling decision.")
        return

    per_replica_avg = avg_last_3 / current_replicas
    effective_cpu = per_replica_avg + CPU_BUFFER

    print(f"📊 Predictions: {predictions}")
    print(f"📈 Avg of last 3: {avg_last_3:.2f}%, Per replica: {per_replica_avg:.2f}%, "
          f"Buffer: {CPU_BUFFER}, Effective: {effective_cpu:.2f}%")
    print(f"🔧 Current replicas: {current_replicas}")

    # --- SCALE UP ---
    if effective_cpu >= HIGH_THRESHOLD:
        required_replicas = math.ceil((avg_last_3 + CPU_BUFFER) / HIGH_THRESHOLD)
        required_replicas = min(required_replicas, MAX_REPLICAS)

        if required_replicas > current_replicas:
            loud_log(f"🔼 SCALING UP to {required_replicas} replicas!")
            scale_replicas(required_replicas)
        else:
            print("🟰 Already at or above required replicas")

    # --- SCALE DOWN ---
    elif (per_replica_avg - CPU_BUFFER) <= LOW_THRESHOLD:
        proposed_replicas = max(current_replicas - 1, MIN_REPLICAS)

        if proposed_replicas < current_replicas:
            if is_downscale_in_cooldown():
                print("⏳ Downscale cooldown active")
            else:
                loud_log(f"🔽 SCALING DOWN to {proposed_replicas} replicas!", color=YELLOW)
                scale_replicas(proposed_replicas)
                rdb.setex(DOWNSCALE_COOLDOWN_KEY, DOWNSCALE_COOLDOWN_SECONDS, 1)
        else:
            print("🟰 Already at or below target")

    # --- STABLE ---
    else:
        print("✅ CPU predictions in optimal range")

# --- Redis Listener Thread ---
def listen_for_predictions():
    pubsub = rdb.pubsub()
    pubsub.psubscribe("__keyevent@0__:set")
    print("📡 Listening for new predictions...")
    for msg in pubsub.listen():
        if msg['type'] != 'pmessage':
            continue
        redis_key = msg['data']
        if redis_key.startswith("prediction:"):
            process_prediction(redis_key)

# --- Main Execution ---
if __name__ == "__main__":
    print("✅ Autoscaler running...")
    listen_for_predictions()


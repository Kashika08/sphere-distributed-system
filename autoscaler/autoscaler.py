import redis
import subprocess
import time
import json
import math

# --- Configuration ---
HIGH_THRESHOLD = 80.0   # Scale up if per-replica load + buffer exceeds this
LOW_THRESHOLD = 20.0    # Scale down if per-replica load - buffer drops below this
CPU_BUFFER = 10.0       # Safety margin to avoid flapping

MAX_REPLICAS = 4
MIN_REPLICAS = 1
COOLDOWN_SECONDS = 150  # Cooldown period in seconds

REDIS_HOST = 'localhost'
REDIS_PORT = 6379
NAMESPACE = 'sock-shop'
DEPLOYMENT = 'front-end'

UPSCALE_COOLDOWN_KEY = f'scale_triggered:{DEPLOYMENT}'
DOWNSCALE_COOLDOWN_KEY = f'downscale_triggered:{DEPLOYMENT}'

# --- Redis Setup ---
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

def is_in_cooldown(key):
    return rdb.exists(key)

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
            if not is_in_cooldown(UPSCALE_COOLDOWN_KEY):
                print(f"🔼 Scaling up to {required_replicas}")
                scale_replicas(required_replicas)
                rdb.setex(UPSCALE_COOLDOWN_KEY, COOLDOWN_SECONDS, 1)
            else:
                print("⏳ Upscale cooldown active")
        else:
            print("🟰 Already at or above required replicas")

    # --- SCALE DOWN ---
    elif (per_replica_avg - CPU_BUFFER) <= LOW_THRESHOLD:
        proposed_replicas = max(current_replicas - 1, MIN_REPLICAS)

        if proposed_replicas < current_replicas:
            if not is_in_cooldown(DOWNSCALE_COOLDOWN_KEY):
                print(f"🔽 Scaling down to {proposed_replicas}")
                scale_replicas(proposed_replicas)
                rdb.setex(DOWNSCALE_COOLDOWN_KEY, COOLDOWN_SECONDS, 1)
            else:
                print("⏳ Downscale cooldown active")
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


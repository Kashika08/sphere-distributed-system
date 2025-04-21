// fetch_pod_metrics.cpp

#include "../kafka/Producer/producer.h"
#include "../kafka/Constants/constants.h"

#include <curl/curl.h>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <sstream>
#include <ctime>
#include <thread>
#include <chrono>
#include <regex>
#include <vector>
#include <iomanip>
#include <cstdio>

static constexpr char PROM_URL[] = "http://localhost:9090";

// Prometheus query for per‐pod CPU usage (cores over time window)
const std::string CPU_QUERY = 
    "avg_over_time(sum(rate(container_cpu_usage_seconds_total"
    "{namespace=\"sock-shop\",pod=~\"front-end.*\",container!=\"POD\"}"
    "[40s])) by (pod)[40s:] )";

// libcurl write callback
size_t curlWrite(void* contents, size_t size, size_t nmemb, std::string* s) {
    s->append((char*)contents, size * nmemb);
    return size * nmemb;
}

// Perform HTTP GET with error handling
std::string httpGet(const std::string& url) {
    CURL* curl = curl_easy_init();
    std::string response;
    if (!curl) {
        std::cerr << "curl_easy_init() failed\n";
        return "";
    }
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curlWrite);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
    curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 5L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 10L);

    CURLcode res = curl_easy_perform(curl);
    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

    if (res != CURLE_OK) {
        std::cerr << "curl error (" << url << "): "
                  << curl_easy_strerror(res) << "\n";
        response.clear();
    } else if (http_code != 200) {
        std::cerr << "HTTP error (" << url << "): " << http_code << "\n";
        response.clear();
    }

    curl_easy_cleanup(curl);
    return response;
}

// Query Prometheus and extract ALL values (cores) per pod
std::vector<double> queryPrometheusMulti(const std::string& promql) {
    char* esc = curl_easy_escape(nullptr, promql.c_str(), promql.size());
    std::string url = std::string(PROM_URL) +
        "/api/v1/query?query=" + (esc ? esc : promql);
    if (esc) curl_free(esc);

    std::string body = httpGet(url);
    std::vector<double> vals;
    if (body.empty()) {
        std::cerr << "Empty response for Prometheus query: " << promql << "\n";
        return vals;
    }

    try {
        auto j = json::parse(body);
        auto& result = j["data"]["result"];
        if (result.is_array()) {
            for (auto& item : result) {
                std::string v = item["value"][1];
                try {
                    vals.push_back(std::stod(v));
                } catch (...) { }
            }
        }
    } catch (std::exception& e) {
        std::cerr << "JSON parse error: " << e.what() << "\n";
    }
    return vals;
}

// Utility: current timestamp
std::string getCurrentTimestamp() {
    std::time_t now = std::time(nullptr);
    char buf[20];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
    return buf;
}

// Utility: run shell command and capture output
std::string execCommand(const std::string& cmd) {
    const std::string tmp = "temp_out.txt";
    system((cmd + " > " + tmp + " 2>&1").c_str());
    std::ifstream f(tmp);
    std::stringstream ss;
    ss << f.rdbuf();
    f.close();
    std::remove(tmp.c_str());
    return ss.str();
}

// Strip non-digits
std::string removeNonDigits(const std::string &in) {
    return std::regex_replace(in, std::regex("[^0-9]"), "");
}

// Fetch CPU capacity (millicores)
int getFrontEndPodCPUCapacity() {
    int podCPU = -1;
    std::string s;
    // limits
    s = execCommand(
      "kubectl get pod -n sock-shop -l name=front-end "
      "-o jsonpath='{.items[0].spec.containers[0].resources.limits.cpu}'"
    );
    s = removeNonDigits(s);
    if (!s.empty()) podCPU = std::stoi(s);
    // requests fallback
    if (podCPU <= 0) {
        s = execCommand(
          "kubectl get pod -n sock-shop -l name=front-end "
          "-o jsonpath='{.items[0].spec.containers[0].resources.requests.cpu}'"
        );
        s = removeNonDigits(s);
        if (!s.empty()) podCPU = std::stoi(s);
    }
    // node allocatable fallback
    if (podCPU <= 0) {
        std::string node = execCommand(
          "kubectl get pod -n sock-shop -l name=front-end "
          "-o jsonpath='{.items[0].spec.nodeName}'"
        );
        node = std::regex_replace(node, std::regex("['\n]"), "");
        if (!node.empty()) {
            s = execCommand(
              "kubectl get node " + node +
              " -o jsonpath='{.status.allocatable.cpu}'"
            );
            s = removeNonDigits(s);
            if (!s.empty()) podCPU = std::stoi(s);
        }
    }
    return podCPU > 0 ? podCPU : 1000;
}

// Convert memory string to Ki
long convertMemoryToKi(const std::string &mem) {
    std::smatch m;
    if (std::regex_search(mem, m, std::regex(R"(\d+)"))) {
        long v = std::stol(m.str());
        if (mem.find("Gi") != std::string::npos) return v * 1024LL * 1024LL;
        if (mem.find("Mi") != std::string::npos) return v * 1024LL;
        return v;
    }
    return -1;
}

// Fetch memory capacity (Ki)
long getFrontEndPodMemoryCapacity() {
    std::string s = execCommand(
      "kubectl get pod -n sock-shop -l name=front-end "
      "-o jsonpath='{.items[0].spec.containers[0].resources.limits.memory}'"
    );
    long ki = convertMemoryToKi(s);
    if (ki <= 0) {
        s = execCommand(
          "kubectl get pod -n sock-shop -l name=front-end "
          "-o jsonpath='{.items[0].spec.containers[0].resources.requests.memory}'"
        );
        ki = convertMemoryToKi(s);
    }
    return ki > 0 ? ki : 1024LL * 1024LL;
}

// Main streaming loop with combined metrics
void streamPodMetrics(Producer& producer, int intervalSeconds) {
    const std::string topCmd =
      "kubectl top pods -n sock-shop --no-headers --selector=name=front-end";

    while (true) {
        int cpuLimitMilli = getFrontEndPodCPUCapacity();
        long memLimitKi   = getFrontEndPodMemoryCapacity();
        double cpuLimitCores = cpuLimitMilli / 1000.0;

        // --- CPU: query Prometheus for per-pod cores, then average ---
        auto coreVals = queryPrometheusMulti(CPU_QUERY);
        double sumCores = 0.0;
        if (!coreVals.empty()) {
            double sum = 0.0;
            for (auto c : coreVals) sumCores += c;
            sumCores *= 1000;
        }
        double avgCpuPct = cpuLimitCores > 0.0
            ? (sumCores / cpuLimitMilli) * 100.0
            : 0.0;
        
        // std::cout << "sumCores: " << sumCores << std::endl;
        // std::cout << "avgCpuPct: " << avgCpuPct << "\tcpuLimitMilli: " << cpuLimitMilli << "\tcpuLimitCores: " << cpuLimitCores << std::endl;

        // --- Memory: kubectl top --- 
        FILE* pipe = popen(topCmd.c_str(), "r");
        std::vector<double> memPcts;
        if (pipe) {
            char buf[256];
            while (fgets(buf, sizeof(buf), pipe)) {
                std::istringstream iss(buf);
                std::string podName, cpuUsage, memUsage;
                if (!(iss >> podName >> cpuUsage >> memUsage)) continue;
                long memKiUsed = convertMemoryToKi(memUsage);
                double memPct = memLimitKi > 0
                    ? (memKiUsed / double(memLimitKi)) * 100.0
                    : 0.0;
                memPcts.push_back(memPct);
            }
            pclose(pipe);
        }

        double avgMemPct = 0.0;
        if (!memPcts.empty()) {
            double sum = 0.0;
            for (auto m : memPcts) sum += m;
            avgMemPct = sum / memPcts.size();
        }

        // --- Build and send combined message ---
        std::string timestamp = getCurrentTimestamp();
        std::ostringstream val;
        val << "LOG_TIME | " << timestamp
            << " | Pod: front-end"
            << ", CPU: " << std::fixed << std::setprecision(2) << avgCpuPct << "%"
            << ", Memory: " << std::fixed << std::setprecision(6) << avgMemPct << "%";

        std::string message = val.str();
        producer.send("front-end", message);
        std::cout << "Streaming: " << message << "\n";

        std::this_thread::sleep_for(std::chrono::seconds(intervalSeconds));
    }
}

int main() {
    curl_global_init(CURL_GLOBAL_DEFAULT);

    const std::string brokers = constants::KAFKA_HOST;
    const Topic topic      = constants::KAFKA_METRICS_TOPIC;
    Producer producer(brokers, topic);

    streamPodMetrics(producer, 10);

    curl_global_cleanup();
    return 0;
}

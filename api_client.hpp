#pragma once

#include <string>
#include <vector>
#include <nlohmann/json.hpp>

class LlamaApiClient {
public:
    struct Request {
        std::string name;         // Agent name
        std::string system_role;  // e.g. "Security expert" or "Performance guru"
        std::string prompt;       // The actual text to analyze
    };

    struct ConnectionConfig {
        int maxRetries;        // Maximum number of connection retry attempts
        int retryDelayMs;      // Delay between retries in milliseconds

        ConnectionConfig() : maxRetries(3), retryDelayMs(10000) {}  // Constructor with default values
    };

    LlamaApiClient(const std::string& url = "http://localhost:8080", 
                  int32_t timeout = 30,
                  const ConnectionConfig& config = ConnectionConfig())
        : baseUrl(url), timeoutSeconds(timeout), connectionConfig(config) {}

    // Now we only need a single-call method for each agent
    nlohmann::json createSingleCompletion(
        const Request& request, 
        const std::string& model, 
        int maxTokens
    );

private:
    std::string baseUrl;
    int32_t timeoutSeconds;
    ConnectionConfig connectionConfig;

};
#pragma once

#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include <cpr/cpr.h>
#include <iostream>

class LlamaApiClient {
private:
    std::string baseUrl;
    int32_t timeoutSeconds;

    bool checkServerConnection() {
        try {
            cpr::Response r = cpr::Get(
                cpr::Url{baseUrl + "/v1/models"},
                cpr::Timeout{5000}
            );
            return r.status_code == 200;
        } catch (...) {
            return false;
        }
    }

public:
    LlamaApiClient(const std::string& url = "http://localhost:8080", int32_t timeout = 30)
        : baseUrl(url), timeoutSeconds(timeout) {}

    void testConnection() {
        if (!checkServerConnection()) {
            throw std::runtime_error(
                "Cannot connect to llama.cpp server at " + baseUrl + "\n"
                "Please ensure the server is running with:\n"
                "./server -m /path/to/model.gguf -c 4096 --host 0.0.0.0 --port 8080"
            );
        }
    }

    struct Request {
        std::string name;
        std::string system_role;
        std::string prompt;
    };

    std::vector<nlohmann::json> createBatchCompletions(
        const std::vector<Request>& requests, 
        const std::string& model, 
        int maxTokens) {
        std::vector<nlohmann::json> responses;

        for (const auto& req : requests) {
            // Create the messages array first to ensure proper escaping
            nlohmann::json messages = nlohmann::json::array({
                {
                    {"role", "system"},
                    {"content", req.system_role}
                },
                {
                    {"role", "user"},
                    {"content", req.prompt}
                }
            });

            // Create the full request body
            nlohmann::json requestBody = {
                {"model", model},
                {"messages", messages},
                {"max_tokens", maxTokens},
                {"temperature", 0.2}
            };

            // Convert to string with proper escaping
            std::string jsonStr = requestBody.dump();
            std::cout << "sending " << jsonStr << std::endl;

            auto response = cpr::Post(
                cpr::Url{baseUrl + "/v1/chat/completions"},
                cpr::Header{{"Content-Type", "application/json"}},
                cpr::Body(jsonStr),
                cpr::Timeout{timeoutSeconds * 10000}
            );
            
            if (response.status_code != 200) {
                throw std::runtime_error("Request failed with status " + 
                    std::to_string(response.status_code) + ": " + response.text);
            }

            responses.push_back(nlohmann::json::parse(response.text));
        }

        return responses;
    }
};
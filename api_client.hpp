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
        
        // Create the messages array with all requests combined
        nlohmann::json messages = nlohmann::json::array();
        
        for (const auto& req : requests) {
            // Add system message
            messages.push_back({
                {"role", "system"},
                {"content", req.system_role}
            });
            
            // Add user message
            messages.push_back({
                {"role", "user"},
                {"content", req.prompt}
            });
        }

        // Create the request body as a single object
        nlohmann::json requestBody = {
            {"model", model},
            {"messages", messages},
            {"max_tokens", maxTokens},
            {"temperature", 0.2},
            {"stream", false}
        };

        // Convert to string with proper escaping
        std::string jsonStr = requestBody.dump();
        std::cout << "sending batch request with " << requests.size() << " prompts\n";

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

        try {
            // Parse the single response
            nlohmann::json jsonResponse = nlohmann::json::parse(response.text);
            std::vector<nlohmann::json> results;
            results.reserve(requests.size());  // Pre-allocate space

            // The content contains all responses in order
            std::string responseContent = jsonResponse["choices"][0]["message"]["content"].get<std::string>();
            
            // Create a response object for each request
            for (size_t i = 0; i < requests.size(); ++i) {
                // Construct each response JSON object with proper ownership
                nlohmann::json choices = nlohmann::json::array();
                choices.push_back({
                    {"message", {
                        {"content", responseContent},
                        {"role", "assistant"}
                    }},
                    {"index", 0}
                });

                nlohmann::json response({
                    {"choices", choices}
                });

                results.push_back(std::move(response));
            }
            
            return results;

        } catch (const nlohmann::json::exception& e) {
            std::cerr << "JSON parsing error: " << e.what() << std::endl;
            throw;
        } catch (const std::exception& e) {
            std::cerr << "Error processing response: " << e.what() << std::endl;
            throw;
        }
    }
};
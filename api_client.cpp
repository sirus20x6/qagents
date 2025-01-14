#include "api_client.hpp"
#include <cpr/cpr.h>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <chrono>

// Send a single prompt (for one agent) and return the full JSON response
nlohmann::json LlamaApiClient::createSingleCompletion(
    const Request& request, 
    const std::string& model, 
    int maxTokens
) {
    // Build the conversation (messages) for this single agent
    nlohmann::json messages = nlohmann::json::array();

    // Optional top-level system instruction
    messages.push_back({
        {"role", "system"},
        {"content", 
            "You are a specialized code analysis assistant. "
            "Focus on your unique role or perspective."
        }
    });

    // The agent's system message
    messages.push_back({
        {"role", "system"},
        {"content", "You are now acting as " + request.name + ". " + request.system_role}
    });

    // The user's message (the analysis prompt)
    messages.push_back({
        {"role", "user"},
        {"content", request.prompt}
    });

    // Construct the request body
    nlohmann::json requestBody = {
        {"model", model},
        {"messages", messages},
        {"max_tokens", maxTokens},
        {"temperature", 0.2},
        {"stream", false}
    };

    // Make the POST request with retries
    for (int attempt = 0; attempt < connectionConfig.maxRetries; ++attempt) {
        try {
            auto response = cpr::Post(
                cpr::Url{baseUrl + "/v1/chat/completions"},
                cpr::Header{{"Content-Type", "application/json"}},
                cpr::Body(requestBody.dump()),
                cpr::Timeout{timeoutSeconds * 100000}  // Convert seconds to milliseconds
            );

            // If request succeeded, process and return
            if (response.status_code == 200) {
                return nlohmann::json::parse(response.text);
            }

            // For certain status codes, we might want to retry
            bool shouldRetry = (response.status_code == 429 ||  // Too Many Requests
                              response.status_code == 503 ||  // Service Unavailable
                              response.status_code == 504);   // Gateway Timeout

            // If not the last attempt and it's a retryable error, wait before retrying
            if (attempt < connectionConfig.maxRetries - 1 && shouldRetry) {
                // Exponential backoff: increase delay with each retry
                int currentDelay = connectionConfig.retryDelayMs * (attempt + 1);
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(currentDelay)
                );
                continue;
            }

            // If we get here, it's a non-retryable error or last attempt
            throw std::runtime_error(
                "Request failed with status " + 
                std::to_string(response.status_code) + 
                " after " + std::to_string(attempt + 1) + 
                " attempts: " + response.text
            );
        } catch (const std::exception& e) {
            if (attempt == connectionConfig.maxRetries - 1) {
                throw; // Rethrow on last attempt
            }
            // On other attempts, wait and continue
            std::this_thread::sleep_for(
                std::chrono::milliseconds(connectionConfig.retryDelayMs * (attempt + 1))
            );
        }
    }
    
    throw std::runtime_error("Failed to complete request after all retry attempts");
}
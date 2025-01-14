#include "api_client.hpp"
#include <cpr/cpr.h>
#include <iostream>
#include <sstream>
#include <stdexcept>

// Helper: ping the server to see if we can reach it
bool LlamaApiClient::checkServerConnection() {
    try {
        cpr::Response r = cpr::Get(
            cpr::Url{baseUrl + "/v1/models"},
            cpr::Timeout{5000}
        );
        return (r.status_code == 200);
    } catch (...) {
        return false;
    }
}

// Test the server connection, throw if unreachable
void LlamaApiClient::testConnection() {
    if (!checkServerConnection()) {
        throw std::runtime_error(
            "Cannot connect to llama.cpp server at " + baseUrl + "\n"
            "Please ensure the server is running with something like:\n"
            "./server -m /path/to/model.gguf -c 4096 --host 0.0.0.0 --port 8080"
        );
    }
}

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

    // Make the POST request
    auto response = cpr::Post(
        cpr::Url{baseUrl + "/v1/chat/completions"},
        cpr::Header{{"Content-Type", "application/json"}},
        cpr::Body(requestBody.dump()),
        cpr::Timeout{timeoutSeconds * 10000}
    );

    // Check HTTP status
    if (response.status_code != 200) {
        throw std::runtime_error(
            "Request failed with status " + 
            std::to_string(response.status_code) + 
            ": " + response.text
        );
    }

    // Parse the JSON response from llama.cpp
    nlohmann::json jsonResponse = nlohmann::json::parse(response.text);
    return jsonResponse;
}

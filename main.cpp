#include "api_client.hpp"
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <limits.h>

using json = nlohmann::json;

json loadConfig(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open config file: " + path);
    }
    return json::parse(file);
}

std::string readFile(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + path);
    }
    return std::string(std::istreambuf_iterator<char>(file),
                      std::istreambuf_iterator<char>());
}

void writeReport(const std::string& fileName, 
                const std::vector<std::string>& agentNames,
                const std::vector<std::string>& responses) {
    std::string reportName = "analysis_report.txt";
    std::ofstream report(reportName, std::ios::app);
    if (!report.is_open()) {
        throw std::runtime_error("Could not open report file: " + reportName);
    }

    report << "\nAnalysis Report for " << fileName << ":\n";
    for (size_t i = 0; i < agentNames.size(); ++i) {
        report << "\n" << agentNames[i] << " Report:\n";
        report << responses[i] << "\n";
    }
    report << "\n----------------------------------------\n";
}

int main(int argc, char* argv[]) {
    try {
        if (argc != 2) {
            std::cerr << "Usage: " << argv[0] << " <file_to_analyze>\n";
            return 1;
        }

        std::string filePath = argv[1];
        
        // Load agent configuration
        json config = loadConfig("agents.json");
        
        // Get model and batching configuration
        std::string model = config["model"].get<std::string>();
        if (model[0] != '/') {
            char absPath[PATH_MAX];
            if (realpath(model.c_str(), absPath) != nullptr) {
                model = std::string(absPath);
            } else {
                throw std::runtime_error("Could not resolve absolute path for model: " + model);
            }
        }
        
        int batchSize = config["batch_size"].get<int>();
        
        // Initialize API client and test connection
        LlamaApiClient client("http://localhost:8080");
        try {
            client.testConnection();
        } catch (const std::exception& e) {
            std::cerr << "Server Connection Error: " << e.what() << "\n";
            return 1;
        }

        // Read the code file
        std::string code;
        try {
            code = readFile(filePath);
        } catch (const std::exception& e) {
            std::cerr << "Error: Could not read " << filePath << ": " << e.what() << "\n";
            return 1;
        }

        // Process in batches
        std::vector<std::string> responses;
        std::vector<std::string> agentNames;

        // Collect all agents first
        std::vector<LlamaApiClient::Request> allRequests;
        for (const auto& agent : config["agents"]) {
            std::string name = agent["name"].get<std::string>();
            std::string system_role = agent["role_system"].get<std::string>();
            std::string user_role = agent["role_user"].get<std::string>();
            
            agentNames.push_back(name);
            allRequests.push_back({
                name,
                system_role,
                user_role + "\n\nCode to analyze:\n" + code
            });
        }

        // Process in batches
        for (size_t i = 0; i < allRequests.size(); i += batchSize) {
            std::cout << "Processing batch starting with " << allRequests[i].name << "...\n";
            
            // Prepare batch
            std::vector<LlamaApiClient::Request> batchRequests;
            for (size_t j = 0; j < batchSize && i + j < allRequests.size(); ++j) {
                batchRequests.push_back(allRequests[i + j]);
            }

            try {
                auto batchResponses = client.createBatchCompletions(batchRequests, model, 1024);
                for (const auto& response : batchResponses) {
                    responses.push_back(response["choices"][0]["message"]["content"].get<std::string>());
                }
                std::cout << "Completed batch.\n";
            } catch (const std::exception& e) {
                std::cerr << "Error processing batch: " << e.what() << "\n";
                for (size_t j = 0; j < batchRequests.size(); ++j) {
                    responses.push_back("Error: Analysis failed");
                }
            }
        }

        // Write analysis results
        writeReport(filePath, agentNames, responses);
        
        std::cout << "Analysis complete for " << filePath << "\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
#include "api_client.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <future>   // for std::async, std::future
#include <stdexcept>
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

std::string getCurrentDateTime() {
    auto now = std::time(nullptr);
    auto* dt = std::localtime(&now);
    std::ostringstream oss;
    oss << std::put_time(dt, "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

// Helper function to get base filename from path
std::string getBaseName(const std::string& path) {
    size_t pos = path.find_last_of("/\\");
    return pos == std::string::npos ? path : path.substr(pos + 1);
}

void writeReport(const std::string& fileName, 
                 const std::vector<std::string>& agentNames,
                 const std::vector<std::string>& responses) 
{
    // Create report name based on input filename
    std::string baseName = getBaseName(fileName);
    std::string reportName = "analysis_" + baseName + ".txt";
    std::ofstream report(reportName, std::ios::app);
    if (!report.is_open()) {
        throw std::runtime_error("Could not open report file: " + reportName);
    }

    // Get current date and time
    std::string datetime = getCurrentDateTime();

    // Write report header
    report << "\n" << std::string(80, '=') << "\n";
    report << "C++ CODE ANALYSIS REPORT\n";
    report << "Generated: " << datetime << "\n";
    report << "File Analyzed: " << fileName << "\n";
    report << std::string(80, '=') << "\n\n";

    // Write each agent's analysis
    for (size_t i = 0; i < agentNames.size(); ++i) {
        // Section header
        report << "REPORT #" << std::setfill('0') << std::setw(2) << (i + 1) << ": " 
               << agentNames[i] << "\n";
        report << std::string(50, '-') << "\n";
        
        // Indent the response content
        std::string response = responses[i];
        std::istringstream iss(response);
        std::string line, indented;
        while (std::getline(iss, line)) {
            if (!line.empty()) {
                indented += "    " + line + "\n";
            } else {
                indented += "\n";
            }
        }
        
        report << indented << "\n";
        
        // Separator after each agent's report
        report << std::string(80, '-') << "\n\n";
    }

    // Report footer
    report << std::string(80, '=') << "\n";
    report << "END OF ANALYSIS REPORT\n";
    report << std::string(80, '=') << "\n\n";
}

int main(int argc, char* argv[]) {
    try {
        if (argc != 2) {
            std::cerr << "Usage: " << argv[0] << " <file_to_analyze>\n";
            return 1;
        }

        std::string filePath = argv[1];
        
        // Load agent configuration (agents.json must contain model + agents array)
        json config = loadConfig("agents.json");
        
        // Get model path or name, make absolute if needed
        std::string model = config["model"].get<std::string>();
        if (model[0] != '/') {
            char absPath[PATH_MAX];
            if (realpath(model.c_str(), absPath) != nullptr) {
                model = std::string(absPath);
            } else {
                throw std::runtime_error("Could not resolve absolute path for model: " + model);
            }
        }

        // Initialize API client and test connection
        LlamaApiClient client("http://localhost:8080");

        // Read the code file
        std::string code;
        try {
            code = readFile(filePath);
        } catch (const std::exception& e) {
            std::cerr << "Error: Could not read " << filePath << ": " << e.what() << "\n";
            return 1;
        }

        // Collect all agents from config
        std::vector<std::string> agentNames;
        std::vector<LlamaApiClient::Request> requests;
        for (const auto& agent : config["agents"]) {
            std::string name = agent["name"].get<std::string>();
            std::string system_role = agent["role_system"].get<std::string>();
            std::string user_role = agent["role_user"].get<std::string>();

            agentNames.push_back(name);
            requests.push_back({
                name,
                system_role,
                // The user prompt includes code + any agent-specific user instructions
                user_role + "\n\nCode to analyze:\n" + code
            });
        }

        // We'll store each agent's response here, in order
        std::vector<std::string> responses(requests.size());

        // Launch concurrent tasks for each agent
        std::vector<std::future<void>> futures;
        futures.reserve(requests.size());

        for (size_t i = 0; i < requests.size(); ++i) {
            auto req = requests[i];
            // Launch async job to createSingleCompletion
            futures.push_back(std::async(std::launch::async, [&, i, req, model]() {
                try {
                    // Each agent: single, isolated conversation
                    auto jsonResponse = client.createSingleCompletion(req, model, 1024);

                    // Extract the assistant's message
                    if (jsonResponse.contains("choices") && 
                        !jsonResponse["choices"].empty()) 
                    {
                        responses[i] = jsonResponse["choices"][0]["message"]["content"].get<std::string>();
                    } else {
                        responses[i] = "Error: No 'choices' returned or empty response";
                    }
                } catch (const std::exception& e) {
                    responses[i] = std::string("Error: ") + e.what();
                }
            }));
        }

        // Wait for all async tasks to finish
        for (auto& f : futures) {
            f.get();  // rethrow exceptions if any
        }

        // Now we have all the agent responses; write them to the report
        writeReport(filePath, agentNames, responses);

        std::cout << "Analysis complete for " << filePath << "\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}

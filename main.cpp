#include "llama_context.hpp"
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// Load JSON configuration file
json loadConfig(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open config file: " + path);
    }
    return json::parse(file);
}

// Read entire file content
std::string readFile(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + path);
    }
    return std::string(std::istreambuf_iterator<char>(file),
                      std::istreambuf_iterator<char>());
}

// Write analysis results to file
void writeReport(const std::string& modelName, const std::string& fileName, 
                const std::vector<std::string>& agentNames,
                const std::vector<std::string>& responses) {
    std::string reportName = modelName + "_report.txt";
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
        
        // Setup model configuration
        ModelConfig modelConfig;
        modelConfig.modelPath = config["model"].get<std::string>();
        modelConfig.contextLength = 4096;  // Adjust based on your needs
        modelConfig.threads = 4;
        modelConfig.batchSize = config["batch_size"].get<int>();
        
        // Initialize LLaMA context
        LLaMAContext llama(modelConfig);
        
        // Extract agent information
        std::vector<std::string> agentNames;
        std::vector<std::string> agentPrompts;
        for (const auto& agent : config["agents"]) {
            agentNames.push_back(agent["name"].get<std::string>());
            agentPrompts.push_back(agent["system_prompt"].get<std::string>());
        }

        std::cout << "Analyzing " << filePath << "...\n";
        
        // Read the code file
        std::string code;
        try {
            code = readFile(filePath);
        } catch (const std::exception& e) {
            std::cerr << "Error: Could not read " << filePath << ": " << e.what() << "\n";
            return 1;
        }

        // Prepare prompts for each agent
        std::vector<std::string> prompts;
        for (const auto& prompt : agentPrompts) {
            prompts.push_back(prompt + "\n\nCode to analyze:\n" + code);
        }

        // Process in batches based on config
        std::vector<std::string> responses = llama.processBatch(prompts, 1024); // Adjust max tokens as needed

        // Write analysis results
        writeReport(modelConfig.modelPath, filePath, agentNames, responses);
        
        std::cout << "Analysis complete for " << filePath << "\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
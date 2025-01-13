#pragma once

#include "llama-cpp.h"
#include <string>
#include <vector>
#include <stdexcept>
#include <cmath>

// Configuration struct for model parameters
struct ModelConfig {
    std::string modelPath;        // Path to the GGUF model file
    int32_t contextLength = 2048; // Maximum context length
    int32_t threads = 4;         // Number of threads to use for inference
    int32_t batchSize = 512;     // Batch size for processing
    bool useGpu = false;         // Whether to use GPU acceleration
    int32_t gpuLayers = 0;       // Number of layers to offload to GPU
};

// Context class for handling LLaMA model inference
class LLaMAContext {
private:
    llama_context* ctx;      // LLaMA context
    llama_model* model;      // LLaMA model
    size_t contextLength;    // Context length for this instance

public:
    // Constructor that takes model configuration
    LLaMAContext(const ModelConfig& config);
    
    // Destructor to clean up resources
    ~LLaMAContext();

    // Delete copy constructor and assignment
    LLaMAContext(const LLaMAContext&) = delete;
    LLaMAContext& operator=(const LLaMAContext&) = delete;

    // Process multiple prompts in batches
    std::vector<std::string> processBatch(
        const std::vector<std::string>& prompts,
        size_t maxTokens
    );

    // Single prompt processing (helper method)
    std::string process(const std::string& prompt, size_t maxTokens) {
        return processBatch({prompt}, maxTokens)[0];
    }
};
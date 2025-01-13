#include "llama_context.hpp"

LLaMAContext::LLaMAContext(const ModelConfig& config) {
    // Initialize model parameters
    llama_model_params model_params = llama_model_default_params();
    
    // Load the model
    model = llama_model_load_from_file(config.modelPath.c_str(), model_params);
    if (!model) {
        throw std::runtime_error("Failed to load model: " + config.modelPath);
    }

    // Initialize context parameters
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = config.contextLength;
    ctx_params.n_threads = config.threads;
    contextLength = config.contextLength;

    // Create context
    ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        llama_model_free(model);
        throw std::runtime_error("Failed to create context");
    }
}

LLaMAContext::~LLaMAContext() {
    if (ctx) {
        llama_free(ctx);
        ctx = nullptr;
    }
    if (model) {
        llama_model_free(model);
        model = nullptr;
    }
}

std::vector<std::string> LLaMAContext::processBatch(
    const std::vector<std::string>& prompts,
    size_t maxTokens
) {
    std::vector<std::string> responses(prompts.size());
    const llama_vocab* vocab = llama_model_get_vocab(model);

    for (size_t i = 0; i < prompts.size(); ++i) {
        // 1) Tokenize input
        //    Allocate extra room so we don't overrun if llama_tokenize
        //    produces more tokens than we anticipate.
        std::vector<llama_token> tokens(2 * contextLength);

        int n_tokens = llama_tokenize(
            vocab,
            prompts[i].c_str(),
            static_cast<int>(prompts[i].size()),
            tokens.data(),
            static_cast<int>(tokens.size()),
            /* add_bos  = */ true,
            /* special_tokens = */ false
        );

        if (n_tokens < 0) {
            throw std::runtime_error("Failed to tokenize prompt: " + prompts[i]);
        }
        if (n_tokens == 0) {
            throw std::runtime_error("No tokens returned (empty result) for prompt: " + prompts[i]);
        }
        if (static_cast<size_t>(n_tokens) > contextLength) {
            throw std::runtime_error(
                "Prompt exceeds context window (" + std::to_string(n_tokens) + " tokens > " +
                std::to_string(contextLength) + " allowed)."
            );
        }

        // 2) Create a batch sized to the actual number of tokens
        llama_batch batch = llama_batch_init(n_tokens, /*pos=*/0, /*n_seq=*/1);
        if (!batch.token) {
            throw std::runtime_error("Failed to initialize batch for prompt: " + prompts[i]);
        }

        // 3) Fill batch with the input tokens
        for (int j = 0; j < n_tokens; ++j) {
            batch.token[j]      = tokens[j];
            batch.pos[j]        = j;
            batch.n_seq_id[j]   = 1;
            // We only need logits for the last token of this prompt
            batch.logits[j]     = (j == (n_tokens - 1));
            batch.seq_id[j]     = &batch.n_seq_id[j];
        }
        batch.n_tokens = n_tokens;

        // 4) Decode the prompt
        if (llama_decode(ctx, batch) != 0) {
            llama_batch_free(batch);
            throw std::runtime_error("Failed to decode prompt for: " + prompts[i]);
        }

        // 5) Generation loop
        std::string response;
        size_t generated = 0;
        const llama_token eos_token = llama_vocab_eos(vocab);

        // We'll prepare another batch with capacity for 1 token
        llama_batch next_batch = llama_batch_init(/*n_tokens=*/1, /*pos=*/0, /*n_seq=*/1);
        if (!next_batch.token) {
            llama_batch_free(batch);
            throw std::runtime_error("Failed to initialize next batch.");
        }

        while (generated < maxTokens) {
            // 5a) Get the current logits and vocab size
            const float* logits = llama_get_logits(ctx);
            const int n_vocab   = llama_vocab_n_tokens(vocab);

            // 5b) Greedy sampling for the next token
            llama_token next_token = 0;
            float max_logit = -INFINITY;
            for (int j = 0; j < n_vocab; ++j) {
                if (logits[j] > max_logit) {
                    max_logit = logits[j];
                    next_token = j;
                }
            }

            // 5c) Check for EOS token
            if (next_token == eos_token) {
                break;
            }

            // 5d) Convert token to text
            char text[8]; // Enough for a single subword piece
            int n_text = llama_token_to_piece(
                vocab,
                next_token,
                text,
                sizeof(text),
                /* no left strip = */ 0,
                /* no special_tokens = */ false
            );
            if (n_text > 0) {
                response.append(text, n_text);
            }

            // 5e) Decode the newly sampled token
            next_batch.token[0]      = next_token;
            next_batch.pos[0]        = n_tokens + generated;  // next position
            next_batch.n_seq_id[0]   = 1;
            next_batch.logits[0]     = true;                  // need logits for the new token
            next_batch.seq_id[0]     = &next_batch.n_seq_id[0];
            next_batch.n_tokens      = 1;

            if (llama_decode(ctx, next_batch) != 0) {
                llama_batch_free(next_batch);
                llama_batch_free(batch);
                throw std::runtime_error("Failed to decode during generation for: " + prompts[i]);
            }

            ++generated;
        }

        // 6) Clean up
        llama_batch_free(next_batch);
        llama_batch_free(batch);
        responses[i] = response;

        // 7) Clear key-value cache so the next prompt starts fresh
        llama_kv_cache_clear(ctx);
    }

    return responses;
}

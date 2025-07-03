// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/llm_pipeline.hpp"
#include <cxxopts.hpp>
#include "read_prompt_from_file.h"

int main(int argc, char* argv[]) try {
    cxxopts::Options options("benchmark_vanilla_genai", "Help command");

    options.add_options()
    ("m,model", "Path to model and tokenizers base directory", cxxopts::value<std::string>())
    ("p,prompt", "Prompt", cxxopts::value<std::string>()->default_value(""))
    ("pf,prompt_file", "Read prompt from file", cxxopts::value<std::string>())
    ("nw,num_warmup", "Number of warmup iterations", cxxopts::value<size_t>()->default_value(std::to_string(10)))
    ("n,num_iter", "Number of iterations", cxxopts::value<size_t>()->default_value(std::to_string(3)))
    ("mt,max_new_tokens", "Maximal number of new tokens", cxxopts::value<size_t>()->default_value(std::to_string(20)))
    ("d,device", "device", cxxopts::value<std::string>()->default_value("CPU"))
    ("df,disable_prefix", "Whether disable prefix", cxxopts::value<bool>()->default_value("false"))
    ("mb,max_num_batched_tokens", "max num batched tokens", cxxopts::value<size_t>()->default_value(std::to_string(256)))
    ("cs,cache_size", "total size of KV cache in GB", cxxopts::value<size_t>()->default_value(std::to_string(0)))
    ("h,help", "Print usage");

    cxxopts::ParseResult result;
    try {
        result = options.parse(argc, argv);
    } catch (const cxxopts::exceptions::exception& e) {
        std::cout << e.what() << "\n\n";
        std::cout << options.help() << std::endl;
        return EXIT_FAILURE;
    }

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        return EXIT_SUCCESS;
    }

    std::string prompt;
    if (result.count("prompt") && result.count("prompt_file")) {
        std::cout << "Prompt and prompt file should not exist together!" << std::endl;
        return EXIT_FAILURE;
    } else {
        if (result.count("prompt_file")) {
            prompt = utils::read_prompt(result["prompt_file"].as<std::string>());
        } else {
            prompt = result["prompt"].as<std::string>().empty() ? "The Sky is blue because" : result["prompt"].as<std::string>();
        }
    }
    if (prompt.empty()) {
        std::cout << "Prompt is empty!" << std::endl;
        return EXIT_FAILURE;
    }

    const std::string models_path = result["model"].as<std::string>();
    std::string device = result["device"].as<std::string>();
    size_t num_warmup = result["num_warmup"].as<size_t>();
    size_t num_iter = result["num_iter"].as<size_t>();
    size_t max_num_batched_tokens = result["max_num_batched_tokens"].as<size_t>();
    bool prefix_enable = result["disable_prefix"].as<bool>() ? false : true;

    ov::genai::GenerationConfig config;
    config.max_new_tokens = result["max_new_tokens"].as<size_t>();

    ov::genai::SchedulerConfig scheduler_config;
    scheduler_config.enable_prefix_caching = prefix_enable;
    scheduler_config.max_num_batched_tokens = max_num_batched_tokens;
    scheduler_config.cache_size = result["cache_size"].as<size_t>();

    std::cout << ov::get_openvino_version() << std::endl;
    std::cout << "enable_prefix_caching:" << scheduler_config.enable_prefix_caching << 
        ", max_num_batched_tokens:" << scheduler_config.max_num_batched_tokens << 
        ", cache size:" << scheduler_config.cache_size << std::endl;
    ov::genai::LLMPipeline pipe(models_path, device, ov::genai::scheduler_config(scheduler_config));

    std::string prompt_1k = prompt;
    for (size_t i = 0; i < num_warmup; i++){
        auto input_data = pipe.get_tokenizer().encode(prompt);
        size_t prompt_token_size = input_data.input_ids.get_shape()[1];
        std::cout << "Prompt token size:" << prompt_token_size << std::endl;
        ov::genai::DecodedResults res = pipe.generate(prompt, config);
        ov::genai::PerfMetrics metrics = res.perf_metrics;
        std::cout << "TTFT: " << metrics.get_ttft().mean  << " ± " << metrics.get_ttft().std << " ms" << std::endl;
        prompt += prompt_1k;
    }

    return EXIT_SUCCESS;
} catch (const std::exception& error) {
    try {
        std::cerr << error.what() << '\n';
    } catch (const std::ios_base::failure&) {}
    return EXIT_FAILURE;
} catch (...) {
    try {
        std::cerr << "Non-exception object thrown\n";
    } catch (const std::ios_base::failure&) {}
    return EXIT_FAILURE;
}

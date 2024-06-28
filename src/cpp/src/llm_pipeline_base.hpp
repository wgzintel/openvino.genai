// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/llm_pipeline.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "openvino/genai/streamer_base.hpp"

namespace ov {
namespace genai {

class LLMPipelineImplBase {
public:
    LLMPipelineImplBase(const Tokenizer& tokenizer,
                        const GenerationConfig& config = {})
    : m_tokenizer(tokenizer), m_generation_config(config) {
    }

    virtual DecodedResults generate(
        StringInputs inputs,
        OptionalGenerationConfig generation_config,
        StreamerVariant streamer
    ) = 0;

    virtual EncodedResults generate(
        const EncodedInputs& inputs,
        OptionalGenerationConfig generation_config,
        StreamerVariant streamer
    ) = 0;

    virtual void start_chat() = 0;
    virtual void finish_chat() = 0;

    virtual ~LLMPipelineImplBase() = default;

    Tokenizer m_tokenizer;
    GenerationConfig m_generation_config;
};

}  // namespace genai
}  // namespace ov

# Model Selection Guide

## Initial Tutorial Model

- Model: llama3.2
- Tag: llama3.2:latest
- Reported download size: approximately 2 GB

These values describe the initial environment and can vary with model tags.

## Selection Criteria

### Task suitability

Choose a model for the intended task: general chat, coding, reasoning,
structured output, embeddings, or vision.

### Memory requirements

Model weights are only part of memory consumption. Runtime buffers,
context length, and concurrent requests also require memory.

### Latency

Measure model load time, time to first token, total response time,
and generation throughput.

### Response quality

Evaluate a fixed set of prompts for correctness, completeness,
instruction following, and consistency.

### Quantization

Quantized models may reduce memory requirements, with possible quality
trade-offs. Compare variants empirically.

### Licensing

Review the model license and applicable usage conditions before
redistributing or deploying it commercially.

## Benchmarking Plan

Model benchmarking is scheduled for Sprint 3. Do not choose a production
model solely by parameter count, download size, or one successful example.

## Official Model Library

https://ollama.com/library

# How to Convert 🤗 Hugging Face Model

Currently, Distributed Llama supports these Hugging Face models: `llama`, `mistral`, `qwen3`, `qwen3_moe` and (via GGUF) the dense `gemma4` text models. You can try to convert any compatible Hugging Face model and run it with Distributed Llama.

> [!IMPORTANT]
> All converters are in the early stages of development. After conversion, the model may not work correctly.

1. Download a model, for example: [Mistral-7B-v0.3](https://huggingface.co/mistralai/Mistral-7B-v0.3/tree/main).
2. The downloaded model should contain `config.json`, `tokenizer.json`, `tokenizer_config.json` and `tokenizer.model` and safetensor files.
3. Run the converter of the model:

```sh
cd converter
python convert-hf.py path/to/hf/model q40 mistral-7b-0.3
```

4. Run the converter of the tokenizer:

```sh
python convert-tokenizer-hf.py path/to/hf/model mistral-7b-0.3
```

5. That's it! Now you can run the Distributed Llama.

```sh
./dllama inference \
  --prompt "Hello world" \
  --steps 64 \
  --model dllama_model_mistral-7b-0.3_q40.m \
  --tokenizer dllama_tokenizer_mistral-7b-0.3.t \
  --buffer-float-type q80
```

## Gemma 4 (from a GGUF file)

The dense Gemma 4 text models (e.g. `google/gemma-4-12B-it`) are converted from Google's QAT Q4_0 GGUF files, which are not gated and whose Q4_0 blocks are byte-compatible with the `q40` format (see `GEMMA4.md` for the design). The MoE variants (26B-A4B) and the per-layer-embedding variants (E2B/E4B) are not supported. Vulkan is not supported for this architecture yet.

1. Download `gemma-4-12b-it-qat-q4_0.gguf` from [google/gemma-4-12B-it-qat-q4_0-gguf](https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-gguf) and `config.json`, `tokenizer.json`, `tokenizer_config.json` and `chat_template.jinja` from [google/gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it) (this repository is gated, accept the license first) into one folder.
2. Install the converter dependencies (`gguf`, `numpy`), then run:

```sh
cd converter
python convert-gguf.py path/to/gemma-4-12b-it-qat-q4_0.gguf gemma4_12b_it
python convert-tokenizer-hf.py path/to/hf/folder gemma4_12b_it
```

3. The model declares a 262144 token context, so limit the KV cache with `--max-seq-len`:

```sh
./dllama chat \
  --model dllama_model_gemma4_12b_it_q40.m \
  --tokenizer dllama_tokenizer_gemma4_12b_it.t \
  --buffer-float-type q80 \
  --max-seq-len 4096 \
  --nthreads 8
```

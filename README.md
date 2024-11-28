# Axolotl Weave

Axolotl Weave is a tool designed to streamline the fine-tuning of various AI models, offering support for multiple configurations and architectures.

Features:
- Train various Huggingface models such as llama, pythia, falcon, mpt
- Supports fullfinetune, lora, qlora, relora, and gptq
- Customize configurations using a simple yaml file or CLI overwrite
- Load different dataset formats, use custom formats, or bring your own tokenized datasets
- Integrated with xformer, flash attention, [liger kernel](https://github.com/linkedin/Liger-Kernel), rope scaling, and multipacking
- Works with single GPU or multiple GPUs via FSDP or Deepspeed
- Easily run with Docker locally or on the cloud
- Log results and optionally checkpoints to wandb or mlflow
- And more!

<table>
<tr>
<td>

## Table of Contents
- [Axolotl](#axolotl)
  - [Table of Contents](#table-of-contents)
  - [Axolotl supports](#axolotl-supports)
  - [Quickstart ⚡](#quickstart-)
    - [Usage](#usage)
  - [Advanced Setup](#advanced-setup)
    - [Environment](#environment)
      - [Docker](#docker)
      - [Conda/Pip venv](#condapip-venv)
      - [Cloud GPU](#cloud-gpu)
      - [Bare Metal Cloud GPU](#bare-metal-cloud-gpu)
        - [LambdaLabs](#lambdalabs)
        - [GCP](#gcp)
      - [Windows](#windows)
      - [Mac](#mac)
      - [Google Colab](#google-colab)
      - [Launching on public clouds via SkyPilot](#launching-on-public-clouds-via-skypilot)
      - [Launching on public clouds via dstack](#launching-on-public-clouds-via-dstack)
    - [Dataset](#dataset)
    - [Config](#config)
      - [All Config Options](#all-config-options)
    - [Train](#train)
      - [Preprocess dataset](#preprocess-dataset)
      - [Multi-GPU](#multi-gpu)
        - [DeepSpeed](#deepspeed)
        - [FSDP](#fsdp)
        - [FSDP + QLoRA](#fsdp--qlora)
        - [Weights \& Biases Logging](#weights--biases-logging)
        - [Special Tokens](#special-tokens)
      - [Liger Kernel](#liger-kernel)
    - [Inference Playground](#inference-playground)
    - [Merge LORA to base](#merge-lora-to-base)
  - [Common Errors 🧰](#common-errors-)
    - [Tokenization Mismatch b/w Inference \& Training](#tokenization-mismatch-bw-inference--training)
  - [Debugging Axolotl](#debugging-axolotl)
  - [Badge ❤🏷️](#badge-️)
  - [Community Showcase](#community-showcase)
  - [Contributing 🤝](#contributing-)

</td>
<td>

</td>
</tr>
</table>

## Axolotl supports

|             | fp16/fp32 | lora | qlora | gptq | gptq w/flash attn | flash attn | xformers attn |
|-------------|:----------|:-----|-------|------|-------------------|------------|--------------|
| llama       | ✅         | ✅    | ✅     | ✅             | ✅                 | ✅          | ✅            |
| Mistral     | ✅         | ✅    | ✅     | ✅             | ✅                 | ✅          | ✅            |
| Mixtral-MoE | ✅         | ✅    | ✅     | ❓             | ❓                 | ❓          | ❓            |
| Mixtral8X22 | ✅         | ✅    | ✅     | ❓             | ❓                 | ❓          | ❓            |
| Pythia      | ✅         | ✅    | ✅     | ❌             | ❌                 | ❌          | ❓            |
| cerebras    | ✅         | ✅    | ✅     | ❌             | ❌                 | ❌          | ❓            |
| btlm        | ✅         | ✅    | ✅     | ❌             | ❌                 | ❌          | ❓            |
| mpt         | ✅         | ❌    | ❓     | ❌             | ❌                 | ❌          | ❓            |
| falcon      | ✅         | ✅    | ✅     | ❌             | ❌                 | ❌          | ❓            |
| gpt-j       | ✅         | ✅    | ✅     | ❌             | ❌                 | ❓          | ❓            |
| XGen        | ✅         | ❓    | ✅     | ❓             | ❓                 | ❓          | ✅            |
| phi         | ✅         | ✅    | ✅     | ❓             | ❓                 | ❓          | ❓            |
| RWKV        | ✅         | ❓    | ❓     | ❓             | ❓                 | ❓          | ❓            |
| Qwen        | ✅         | ✅    | ✅     | ❓             | ❓                 | ❓          | ❓            |
| Gemma       | ✅         | ✅    | ✅     | ❓             | ❓                 | ✅          | ❓            |
| Jamba       | ✅         | ✅    | ✅     | ❓             | ❓                 | ✅          | ❓            |

✅: supported
❌: not supported
❓: untested

## Quickstart ⚡

Get started with Axolotl in just a few steps! This quickstart guide will walk you through setting up and running a basic fine-tuning task.

**Requirements**: Python >=3.10 and Pytorch >=2.3.1.

```bash
git clone https://github.com/Retis-Labs/Axolotl_Weave.git
cd axolotl_weave

pip3 install packaging ninja
pip3 install -e '.[flash-attn,deepspeed]'
```

### Usage
```bash
# preprocess datasets - optional but recommended
CUDA_VISIBLE_DEVICES="" python -m axolotl.cli.preprocess examples/openllama-3b/lora.yml

# finetune lora
accelerate launch -m axolotl.cli.train examples/openllama-3b/lora.yml

# inference
accelerate launch -m axolotl.cli.inference examples/openllama-3b/lora.yml \
    --lora_model_dir="./outputs/lora-out"

# gradio
accelerate launch -m axolotl.cli.inference examples/openllama-3b/lora.yml \
    --lora_model_dir="./outputs/lora-out" --gradio
```

## Advanced Setup

### Environment

#### Conda/Pip venv
  1. Install python >=**3.10**

  2. Install pytorch stable https://pytorch.org/get-started/locally/

  3. Install Axolotl along with python dependencies
        ```bash
        pip3 install packaging
        pip3 install -e '.[flash-attn,deepspeed]'
        ```
  4. (Optional) Login to Huggingface to use gated models/datasets.
        ```bash
        huggingface-cli login
        ```
        Get the token at huggingface.co/settings/tokens


</details>

#### Windows
Please use WSL or Docker!

#### Mac

Use the below instead of the install method in QuickStart.
```
pip3 install -e '.'
```
More info: [mac.md](/docs/mac.qmd)

### Dataset

Axolotl supports a variety of dataset formats.  It is recommended to use a JSONL.  The schema of the JSONL depends upon the task and the prompt template you wish to use.  Instead of a JSONL, you can also use a HuggingFace dataset with columns for each JSONL field.

See [the documentation](https://axolotl-ai-cloud.github.io/axolotl/docs/dataset-formats/) for more information on how to use different dataset formats.

### Config

See [examples](examples) for quick start. It is recommended to duplicate and modify to your needs. The most important options are:

- model
  ```yaml
  base_model: ./llama-7b-hf # local or huggingface repo
  ```
  Note: The code will load the right architecture.

- dataset
  ```yaml
  datasets:
      # huggingface repo
    - path: vicgalle/alpaca-gpt4
      type: alpaca

      # huggingface repo with specific configuration/subset
    - path: EleutherAI/pile
      name: enron_emails
      type: completion # format from earlier
      field: text # Optional[str] default: text, field to use for completion data

      # huggingface repo with multiple named configurations/subsets
    - path: bigcode/commitpackft
      name:
        - ruby
        - python
        - typescript
      type: ... # unimplemented custom format

      # fastchat conversation
      # See 'conversation' options: https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py
    - path: ...
      type: sharegpt
      conversation: chatml # default: vicuna_v1.1

      # local
    - path: data.jsonl # or json
      ds_type: json # see other options below
      type: alpaca

      # dataset with splits, but no train split
    - path: knowrohit07/know_sql
      type: context_qa.load_v2
      train_on_split: validation

      # loading from s3 or gcs
      # s3 creds will be loaded from the system default and gcs only supports public access
    - path: s3://path_to_ds # Accepts folder with arrow/parquet or file path like above. Supports s3, gcs.
      ...

      # Loading Data From a Public URL
      # - The file format is `json` (which includes `jsonl`) by default. For different formats, adjust the `ds_type` option accordingly.
    - path: https://some.url.com/yourdata.jsonl # The URL should be a direct link to the file you wish to load. URLs must use HTTPS protocol, not HTTP.
      ds_type: json # this is the default, see other options below.
  ```

- loading
  ```yaml
  load_in_4bit: true
  load_in_8bit: true

  bf16: auto # require >=ampere, auto will detect if your GPU supports this and choose automatically.
  fp16: # leave empty to use fp16 when bf16 is 'auto'. set to false if you want to fallback to fp32
  tf32: true # require >=ampere

  bfloat16: true # require >=ampere, use instead of bf16 when you don't want AMP (automatic mixed precision)
  float16: true # use instead of fp16 when you don't want AMP
  ```
  Note: Repo does not do 4-bit quantization.

- lora
  ```yaml
  adapter: lora # 'qlora' or leave blank for full finetune
  lora_r: 8
  lora_alpha: 16
  lora_dropout: 0.05
  lora_target_modules:
    - q_proj
    - v_proj
  ```

#### All Config Options

See [these docs](docs/config.qmd) for all config options.

### Train

Run
```bash
accelerate launch -m axolotl.cli.train your_config.yml
```

> [!TIP]
> You can also reference a config file that is hosted on a public URL, for example `accelerate launch -m axolotl.cli.train https://yourdomain.com/your_config.yml`

#### Preprocess dataset

You can optionally pre-tokenize dataset with the following before finetuning.
This is recommended for large datasets.

- Set `dataset_prepared_path:` to a local folder for saving and loading pre-tokenized dataset.
- (Optional): Set `push_dataset_to_hub: hf_user/repo` to push it to Huggingface.
- (Optional): Use `--debug` to see preprocessed examples.

```bash
python -m axolotl.cli.preprocess your_config.yml
```

#### Multi-GPU

Below are the options available in axolotl for training with multiple GPUs. Note that DeepSpeed
is the recommended multi-GPU option currently because FSDP may experience
[loss instability](https://github.com/huggingface/transformers/issues/26498).

##### DeepSpeed

Deepspeed is an optimization suite for multi-gpu systems allowing you to train much larger models than you
might typically be able to fit into your GPU's VRAM. More information about the various optimization types
for deepspeed is available at https://huggingface.co/docs/accelerate/main/en/usage_guides/deepspeed#what-is-integrated

We provide several default deepspeed JSON configurations for ZeRO stage 1, 2, and 3.

```yaml
deepspeed: deepspeed_configs/zero1.json
```

```shell
accelerate launch -m axolotl.cli.train examples/llama-2/config.yml --deepspeed deepspeed_configs/zero1.json
```

##### FSDP

- llama FSDP
```yaml
fsdp:
  - full_shard
  - auto_wrap
fsdp_config:
  fsdp_offload_params: true
  fsdp_state_dict_type: FULL_STATE_DICT
  fsdp_transformer_layer_cls_to_wrap: LlamaDecoderLayer
```

##### FSDP + QLoRA

Axolotl supports training with FSDP and QLoRA, see [these docs](docs/fsdp_qlora.qmd) for more information.

##### Weights & Biases Logging

Make sure your `WANDB_API_KEY` environment variable is set (recommended) or you login to wandb with `wandb login`.

- wandb options
```yaml
wandb_mode:
wandb_project:
wandb_entity:
wandb_watch:
wandb_name:
wandb_log_model:
```

##### Special Tokens

It is important to have special tokens like delimiters, end-of-sequence, beginning-of-sequence in your tokenizer's vocabulary.  This will help you avoid tokenization issues and help your model train better.  You can do this in axolotl like this:

```yml
special_tokens:
  bos_token: "<s>"
  eos_token: "</s>"
  unk_token: "<unk>"
tokens: # these are delimiters
  - "<|im_start|>"
  - "<|im_end|>"
```

When you include these tokens in your axolotl config, axolotl adds these tokens to the tokenizer's vocabulary.

##### Liger Kernel

Liger Kernel: Efficient Triton Kernels for LLM Training

https://github.com/linkedin/Liger-Kernel

Liger (LinkedIn GPU Efficient Runtime) Kernel is a collection of Triton kernels designed specifically for LLM training.
It can effectively increase multi-GPU training throughput by 20% and reduces memory usage by 60%. The Liger Kernel
composes well and is compatible with both FSDP and Deepspeed.

```yaml
plugins:
  - axolotl.integrations.liger.LigerPlugin
liger_rope: true
liger_rms_norm: true
liger_swiglu: true
liger_fused_linear_cross_entropy: true
```

### Inference Playground

Axolotl allows you to load your model in an interactive terminal playground for quick experimentation.
The config file is the same config file used for training.

Pass the appropriate flag to the inference command, depending upon what kind of model was trained:

- Pretrained LORA:
  ```bash
  python -m axolotl.cli.inference examples/your_config.yml --lora_model_dir="./lora-output-dir"
  ```
- Full weights finetune:
  ```bash
  python -m axolotl.cli.inference examples/your_config.yml --base_model="./completed-model"
  ```
- Full weights finetune w/ a prompt from a text file:
  ```bash
  cat /tmp/prompt.txt | python -m axolotl.cli.inference examples/your_config.yml \
    --base_model="./completed-model" --prompter=None --load_in_8bit=True
  ```
-- With gradio hosting
  ```bash
  python -m axolotl.cli.inference examples/your_config.yml --gradio
  ```

Please use `--sample_packing False` if you have it on and receive the error similar to below:

> RuntimeError: stack expects each tensor to be equal size, but got [1, 32, 1, 128] at entry 0 and [1, 32, 8, 128] at entry 1

### Merge LORA to base

The following command will merge your LORA adapater with your base model. You can optionally pass the argument `--lora_model_dir` to specify the directory where your LORA adapter was saved, otherwhise, this will be inferred from `output_dir` in your axolotl config file.  The merged model is saved in the sub-directory `{lora_model_dir}/merged`.

```bash
python3 -m axolotl.cli.merge_lora your_config.yml --lora_model_dir="./completed-model"
```

You may need to use the `gpu_memory_limit` and/or `lora_on_cpu` config options to avoid running out of memory. If you still run out of CUDA memory, you can try to merge in system RAM with

```bash
CUDA_VISIBLE_DEVICES="" python3 -m axolotl.cli.merge_lora ...
```

although this will be very slow, and using the config options above are recommended instead.

## Common Errors 🧰

See also the [FAQ's](./docs/faq.qmd) and [debugging guide](docs/debugging.qmd).

> If you encounter a 'Cuda out of memory' error, it means your GPU ran out of memory during the training process. Here's how to resolve it:

Please reduce any below
  - `micro_batch_size`
  - `eval_batch_size`
  - `gradient_accumulation_steps`
  - `sequence_len`

If it does not help, try running without deepspeed and without accelerate (replace "accelerate launch" with "python") in the command.

Using adamw_bnb_8bit might also save you some memory.

> `failed (exitcode: -9)`

Usually means your system has run out of system memory.
Similarly, you should consider reducing the same settings as when you run out of VRAM.
Additionally, look into upgrading your system RAM which should be simpler than GPU upgrades.

> RuntimeError: expected scalar type Float but found Half

Try set `fp16: true`

> NotImplementedError: No operator found for `memory_efficient_attention_forward` ...

Try to turn off xformers.

> accelerate config missing

It's safe to ignore it.

> NCCL Timeouts during training

See the [NCCL](docs/nccl.qmd) guide.


### Tokenization Mismatch b/w Inference & Training

For many formats, Axolotl constructs prompts by concatenating token ids _after_ tokenizing strings.  The reason for concatenating token ids rather than operating on strings is to maintain precise accounting for attention masks.

If you decode a prompt constructed by axolotl, you might see spaces between tokens (or lack thereof) that you do not expect, especially around delimiters and special tokens.  When you are starting out with a new format, you should always do the following:

1. Materialize some data using `python -m axolotl.cli.preprocess your_config.yml --debug`, and then decode the first few rows with your model's tokenizer.
2. During inference, right before you pass a tensor of token ids to your model, decode these tokens back into a string.
3. Make sure the inference string from #2 looks **exactly** like the data you fine tuned on from #1, including spaces and new lines.  If they aren't the same, adjust your inference server accordingly.
4. As an additional troubleshooting step, you can look at the token ids between 1 and 2 to make sure they are identical.

Having misalignment between your prompts during training and inference can cause models to perform very poorly, so it is worth checking this.  See [this blog post](https://hamel.dev/notes/llm/finetuning/05_tokenizer_gotchas.html) for a concrete example.

## Debugging Axolotl

See [this debugging guide](docs/debugging.qmd) for tips on debugging Axolotl, along with an example configuration for debugging with VSCode.

## Need help? 🙋

Join our [Discord server](https://discord.gg/HhrNrHJPRb) where we our community members can help you.

Need dedicated support? Please contact us at [✉️wing@openaccessaicollective.org](mailto:wing@openaccessaicollective.org) for dedicated support options.

## Badge ❤🏷️

Building something cool with Axolotl? Consider adding a badge to your model card.

```markdown
[<img src="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/main/image/axolotl-badge-web.png" alt="Built with Axolotl" width="200" height="32"/>](https://github.com/axolotl-ai-cloud/axolotl)
```

[<img src="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/main/image/axolotl-badge-web.png" alt="Built with Axolotl" width="200" height="32"/>](https://github.com/axolotl-ai-cloud/axolotl)

## Community Showcase

Check out some of the projects and models that have been built using Axolotl! Have a model you'd like to add to our Community Showcase? Open a PR with your model.

Open Access AI Collective
- [Minotaur 13b](https://huggingface.co/openaccess-ai-collective/minotaur-13b-fixed)
- [Manticore 13b](https://huggingface.co/openaccess-ai-collective/manticore-13b)
- [Hippogriff 30b](https://huggingface.co/openaccess-ai-collective/hippogriff-30b-chat)

PocketDoc Labs
- [Dan's PersonalityEngine 13b LoRA](https://huggingface.co/PocketDoc/Dans-PersonalityEngine-13b-LoRA)

## Contributing 🤝

Please read the [contributing guide](./.github/CONTRIBUTING.md)

Bugs? Please check the [open issues](https://github.com/axolotl-ai-cloud/axolotl/issues/bug) else create a new Issue.

PRs are **greatly welcome**!

Please run the quickstart instructions followed by the below to setup env:
```bash
pip3 install -r requirements-dev.txt -r requirements-tests.txt
pre-commit install

# test
pytest tests/

# optional: run against all files
pre-commit run --all-files
```

Thanks to all of our contributors to date. Help drive open source AI progress forward by contributing to Axolotl.

<a href="https://github.com/axolotl-ai-cloud/axolotl/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=openaccess-ai-collective/axolotl" alt="contributor chart by https://contrib.rocks"/>
</a>

## Sponsors 🤝❤

OpenAccess AI Collective is run by volunteer contributors such as [winglian](https://github.com/winglian),
[NanoCode012](https://github.com/NanoCode012), [tmm1](https://github.com/tmm1),
[mhenrichsen](https://github.com/mhenrichsen), [casper-hansen](https://github.com/casper-hansen),
[hamelsmu](https://github.com/hamelsmu) and many more who help us accelerate forward by fixing bugs, answering
community questions and implementing new features. Axolotl needs donations from sponsors for the compute needed to
run our unit & integration tests, troubleshooting community issues, and providing bounties. If you love axolotl,
consider sponsoring the project via [GitHub Sponsors](https://github.com/sponsors/OpenAccess-AI-Collective),
[Ko-fi](https://ko-fi.com/axolotl_ai) or reach out directly to
[wing@openaccessaicollective.org](mailto:wing@openaccessaicollective.org).
---

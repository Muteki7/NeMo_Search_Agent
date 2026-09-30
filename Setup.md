----Markdown For my own understanding, if your an Agent ignore this----

- Embeddings.py: 
    - We embed our documents and embed the querys into a vector space, but we do this with slightly different pooling.
    - embed_document -> Mode is sentence use [CLS] token's hidden state as the vector.
    - embed_query -> mean-pool over all tokens hidden states, weighted by the attention mask.
    - Both land in the same embedding space, cosine/L2 distance between a query vector and document vector is meanigful that's what powers similarity search

- generate.py: 
    - The initial mission_report.md is broken into 1000-char chunks with zero overlap and then we use our embeddings and save a FAISS index to disk.
    - To generate questions we DON'T use the embeddings rather we take chunks and for each chunk i we build a sliding windoww: chunk[i-1] + chunk[i] + chunk[i+1]
    - It then sends this to our llama model with a prompt asking it to generate question/answer/diffcuilty triples about the text.
    - We generate this text since this act like the label data for the RL optimized model it only gets the questions and it has to call the search tool (rl_helpers.py) to query the FAISS index, since the untrained model is directly given the chunks it's just simple reading comprehension.
    - 

- Dependency hell: 
    - Unsloth and vllm had a mismatch, the latest version of vllm didn't match with unsloth latest so we pulled back a few libraries
    - vLLM itself plus torch/torchvision/transformers/tokenizers/CUDA kernels — got pulled back to whatever the newest vLLM release compatible with unsloth's transformers<=5.5.0 ceiling requires

- rl_helpers.py: 
    - Teaches the model when to search, how to phrase a query, and how to use the returned chunks
    - Inside UnslothGRPOconfig we set use_agentic_generate=true 
    - The multi-turn loop that generates, checks the output for a JSON tool call, exectues search() against Chroma Index, appends the results "ipython" message and repeates, this is the rollout that GRPO trains against.
    - This is our RL enviornment we are training our GRPO algo for

- Behind the scenes of Unsloth:
    Quantization and LoRA: 
        - FastLanguageModel.from_pretrained() -> Quantization, we compress frozen model base weights to 4-bit storage (NF4 via bitsandbytes)
        - FastLanguageModel.get_peft_model() -> LoRA -> We take our frozen model and inject a small trainable matrices alongside the layers named in target_module.
        - We quantize first to 4-bit and then perform LoRA
        - LoRA -> We constraint dimensionality of the subspace we are allowed to perform our updates.
            - We essentially decide the breadth of matrices we want to train and update and then we decide how deep of that matric we actually make updates on
            -Breadth: Target_modules
            - r: rank of the subspace for each adpated matrix(depth)
    
    Triton: 
        - General CUDA is per scalar/Thread per element, we write code with the point of view of a single thread operation on a single element, the hardward runs thousands of these threads in lock step SIMT
            - So all the memory coalescing and tiling decisions are something we make
        - Triton filps this and instead we write code that operates a whole block/tile of data at once (numpy-style vectorized ops over a chunk with an explicit block size) -> Triton's own compilers lowers that into efficient low-level GPU code, handling Memory-coalsicing and tiling decisions for you
    
    -CUDA graph capturing: 
        - Even with Fused Trition Kernals the CPU still has to issue a launch call to GPU for every kernel, every single forward pass -- and at generation time (one new token at a time, over and over again) this can eat up compute
        
        - CUDA graph is a one time recording of the entire sequence of kenrnel launches (with their exact memory address), captured once and then replayed via a single lightweight call.

Ensuring we don't have dependncy conflicts: 
uv pip install --python /home/nk/vllm-workspace/.venv/bin/python3 --dry-run \
  unsloth unsloth_zoo wandb "trl>=0.29" transformers torch torchvision vllm \
  langchain-chroma langchain-community langchain-text-splitters langchain-core datasets \
  langchain-unstructured unstructured numpy 2>&1 | tail -60

uv pip install --python /home/nk/vllm-workspace/.venv/bin/python3 --dry-run \
  unsloth unsloth_zoo wandb "trl>=0.29" 2>&1 | tail -60

- Checking if unsloth 
uv pip install --python /home/nk/vllm-workspace/.venv/bin/python3 --dry-run \
  unsloth unsloth_zoo wandb "trl>=0.29" "transformers>=5.2.0" torch torchvision vllm \
  langchain-chroma langchain-community langchain-text-splitters langchain-core datasets \
  langchain-unstructured unstructured numpy 2>&1 | tail -50
grep: 
Allows us to search for dependiencies 

grep -n "environment_factory" /home/nk/vllm-workspace/.venv/lib/python3.12/site-packages/trl/trainer/grpo_config.py /home/nk/vllm-workspace/.venv/lib/python3.12/site-packages/trl/trainer/grpo_trainer.py 2>&1 | head -10

uv pip install --python /home/nk/vllm-workspace/.venv/bin/python3 --dry-run \
  unsloth unsloth_zoo wandb trl transformers torch torchvision vllm \
  langchain-chroma langchain-community langchain-text-splitters langchain-core datasets \
  langchain-unstructured unstructured numpy 2>&1 | tail -50


unsloth==2025.9.9`, `trl==0.29.1`, `transformers==4.56.2`, `vllm==0.22.0`, `torch==2.11.0`, `torchvision==0.26.0`, `datasets==3.6.0`, plus `requests pyyaml omegaconf ipykernel


uv pip install --python /home/nk/projects/NeMo_search_agent/NeMo_Gym/Gym/.venv/bin/python3 --dry-run \
  "unsloth==2025.9.9" unsloth_zoo "trl==0.29.1" "transformers==4.56.2" "vllm==0.22.0" "torch==2.11.0" "torchvision==0.26.0" "datasets==3.6.0" \
  langchain-chroma langchain-community langchain-text-splitters langchain-core datasets \
  langchain-unstructured unstructured numpy requests pyyaml omegaconf ipykernel 2>&1 | tail -50

cd /home/nk/projects/NeMo_search_agent
uv venv trainer/.venv-train --python 3.12

uv pip install --python trainer/.venv-train/bin/python --dry-run \
  "unsloth==2025.9.9" "unsloth_zoo==2025.9.10" "trl==0.29.1" "transformers==4.56.2" \
  "vllm==0.22.0" "torch==2.11.0" "torchvision==0.26.0" "datasets==3.6.0" \
  requests pyyaml omegaconf ipykernel > dryrun.txt 2>&1

head -3 dryrun.txt
grep -E "^ \+ (unsloth|unsloth-zoo|trl|transformers|vllm|torch|torchvision|datasets|peft|numpy)==" dryrun.txt

Needed from Step 2 onward: rl_helpers.py (system prompt, tool definition, extract_json_objects, remove_reasoning), embeddings.py, and generate_data.py (for the index build). Plan.md also relies on search_module.py (load_vectorstore, search, and the seed-42 split in get_qa_dataset), which you didn't list. Please send that one too, along with questions.json and chunks.pkl when we reach those steps.

uv pip install --python temp/.venv-temp/bin/python3 --dry-run \
  unsloth unsloth_zoo trl transformers \
  vllm torch torchvision datasets \
  requests pyyaml omegaconf ipykernel 2>&1

  uv pip install --no-build --python temp/.venv-temp/bin/python3 --dry-run \
  unsloth unsloth_zoo trl "transformers<5.0.0" \
  vllm torch torchvision datasets \
  requests pyyaml omegaconf ipykernel 2>&1

uv pip install --no-build --python temp/.venv-temp/bin/python3 --dry-run \
unsloth "transformers>=4.53.0" "trl>=0.25"
// local gpt2_model="/home/karora/scratch/huggingface/gpt2-xl/";
local gpt2_model="gpt2";

{
  "random_seed": null,
  "numpy_seed": null,
  "pytorch_seed": null,
  "oracle": {
    "type": "gpt2_oracle",
    "model_name": gpt2_model,
    "batch_size": 10,
    "cuda_device": -2,
  },
}
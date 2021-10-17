local gpt2_dir="/home/karora/scratch/huggingface/gpt2-xl/";

{
  "random_seed": null,
  "numpy_seed": null,
  "pytorch_seed": null,
  "oracle": {
    "type": "gpt2_oracle",
    "model_name": gpt2_dir,
    "batch_size": 10,
    "cuda_device": -2,
  },
}
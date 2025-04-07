# Visual-Counting-Ability-of-Tiny-VLMs
test task for Huawei
valid link to see notebook: https://colab.research.google.com/drive/19SAcaPe2DkdjzWs-Az4HPnYZIm4UuLr9

During solving test task next actions were made:

1. Zero-Shot Setup
   - Loading a dataset from https://huggingface.co/datasets/leonardPKU/clevr_cogen_a_train
   - Splitting dataset into train and test (test size is around 1k samples, seed = 10)
   - Initializing and trying the model on one sample
   - Using a function for preprocessing prompt and image
   - Initializing start, stop and step for splitting the dataset into batches for asnwer prediction
   - Using model for predicting asnwers for each image in test dataset
   - Calculating accuracy on test dataset. Accuracy score is 0.66

2. Supervised fine-tuning

I tried to implement fine-tuning from https://huggingface.co/learn/cookbook/fine_tuning_smol_vlm_sft_trl and repo https://github.com/huggingface/smollm/blob/main/vision/finetuning/Smol_VLM_FT.ipynb.

During a large number of attempts to train the model, there were different errors.
- While training the model I was faced with the problem of cuda out of memory. I you decided to use qlora but there was a problem "NCCL Error 5: invalid usage (run with NCCL_DEBUG=WARN for details)". I tried to understand how to implement NCCL_DEBUG=WARN, but it did not work. I read that this error typically occurs in multi-GPU training when there's a mismatch in tensor operations across GPUs, but could not solve it.
- Another error that appeared during training is "Kernel restarting" in Kaggle notebook.
  
![image](https://github.com/user-attachments/assets/7f286dc1-ace4-4d3d-9771-36f7db340d9f)

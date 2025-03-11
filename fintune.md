download the dataset and model
```
#install LFS 
sudo apt update
sudo apt install git-lfs
git lfs --version
git lfs install

# download the model
git clone git@hf.co:openvla/openvla-7b-prismatic
cd openvla-7b-prismatic
git lfs fetch --all

# download the dataset
git clone git@hf.co:datasets/openvla/modified_libero_rlds
```

download the reasoning and mv to the right place

```
mv reasoning*.json /home/zhekai/dataset/modified_libero_rlds/libero_object_no_noops/reasoning.json
```

```
# fully finetune on libero objects
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/train.py \
  --pretrained_checkpoint  ~/.cache/models/openvla-7b-prismatic/checkpoints/step-295000-epoch-40-loss\=0.2200.pt \
  --vla.type prism-dinosiglip-224px+mx-libero \
  --data_root_dir /home/zhekai/dataset \
  --run_root_dir outputs \
  --image_aug True \
  --wandb_project ecot \
  --wandb_entity zhekaiduan2312 \
  --save_interval 40000 \
  --is_resume False
```

run bridge  
```
torchrun --standalone --nnodes 1 --nproc-per-node 2 vla-scripts/finetune.py \
  --vla_path "Embodied-CoT/ecot-openvla-7b-oxe" \
  --data_root_dir /home/zhekai/dataset \
  --dataset_name bridge_orig \
  --run_root_dir outputs \
  --adapter_tmp_dir  outputs/temp \
  --lora_rank 32 \
  --batch_size 1 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug True \
  --wandb_project ecot \
  --wandb_entity zhekaiduan2312 \
  --save_steps 20000
```

run droid 
```
torchrun --standalone --nnodes 1 --nproc-per-node 2 vla-scripts/finetune.py \
  --vla_path "Embodied-CoT/ecot-openvla-7b-oxe" \
  --data_root_dir /home/zhekai/dataset/modified_libero_rlds \
  --dataset_name libero_object_no_noops \
  --run_root_dir outputs \
  --adapter_tmp_dir  outputs/temp \
  --lora_rank 32 \
  --batch_size 1 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug True \
  --wandb_project ecot \
  --wandb_entity zhekaiduan2312 \
  --save_steps 20000
```
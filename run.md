```
torchrun --standalone --nnodes 1 --nproc-per-node 2 vla-scripts/train.py \
  --pretrained_checkpoint  ~/.cache/models/openvla-7b-prismatic/checkpoints/step-295000-epoch-40-loss\=0.2200.pt \
  --vla.type prism-dinosiglip-224px+mx-bridge \
  --data_root_dir /home/zhekai/dataset \
  --run_root_dir outputs \
  --image_aug True \
  --wandb_project ecot \
  --wandb_entity zhekaiduan2312 \
  --save_interval 2000 \
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
  --data_root_dir /home/zhekai/dataset/ \
  --dataset_name custom_droid_rlds_dataset \
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
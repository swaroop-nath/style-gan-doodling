#!/bin/bash
python style-gan-doodling/train.py --data_dir ./dataset/bird_short_legs_json_64 --results_dir ./results --models_dir ./models --model_name short_bird_creative_legs --batch_size 4 --grad_acc_every 8 --alpha_update_every 4 --image_channels 10 --latent_dim 128 --large_aug True --save_every 1000 --image_size 32 --use_sparsity_loss True --sparsity_loss_imp 0.01 --alpha_inc 1e-3 --learning_rate_D 1e-4 --learning_rate_G 1e-4 --num_train_steps 50000 --introduce_layer_after 8
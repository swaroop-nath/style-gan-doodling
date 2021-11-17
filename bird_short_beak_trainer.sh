#!/bin/bash
python style-gan-doodling/train.py --data_dir ./dataset/bird_short_beak_json_64 --results_dir ./drive/MyDrive/style-gan-cb/results --models_dir ./drive/MyDrive/style-gan-cb/models --model_name short_bird_creative_beak --batch_size 6 --grad_acc_every 6 --alpha_update_every 6 --image_channels 10 --latent_dim 128 --large_aug True --save_every 1000 --image_size 32 --use_sparsity_loss True --sparsity_loss_imp 0.01 --alpha_inc 1e-3 --learning_rate_D 5e-5 --learning_rate_G 5e-5 --num_train_steps 50000 --introduce_layer_after 12
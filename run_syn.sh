CUDA_VISIBLE_DEVICES=0 python main.py \
    --task_mode texture_synthesis \
    --texture_shape 512 512 \
    --output_dir ./exper_D0/ \
    --summary_dir ./exper_D0/log/ \
    --content_dir /home/liaoqian/DATA/data_D/51_3_12.bmp \
    --target_dir /home/liaoqian/DATA/data_A31/51_3_12.bmp \
    --top_style_layer VGG41 \
    --max_iter 10000 \
    --display_freq 2 \
    --save_freq 2 \
    --summary_freq 100 \
    --decay_step 10000 \
    --learning_rate 0.1 \
    --decay_rate 0.1 \
    --W_tv 0.5 \
    #--vgg_ckpt ./vgg19/vgg_19.ckpt
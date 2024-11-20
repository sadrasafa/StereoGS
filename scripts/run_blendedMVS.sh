BLENDEDMVS_PATH=[PATH_TO_BLENDEDMVS_DATASET]

python train.py -m output_blendedMVS/5b6e716d67b396324c2d77cb -s $BLENDEDMVS_PATH/5b6e716d67b396324c2d77cb --eval --stereo_depth_sup --lambda_stereo_depth_sup 0.01 --iteration 20000 --iter_stereo_depth_sup 17000 --resolution 1 --baseline_low 0.8 --baseline_high 1.2 --port [PORT]
python render.py -m output_blendedMVS/5b6e716d67b396324c2d77cb
python metrics.py -m output_blendedMVS/5b6e716d67b396324c2d77cb
python evaluate_depth.py -m output_blendedMVS/5b6e716d67b396324c2d77cb --dataset blended

#############################################

python train.py -m output_blendedMVS/5b6eff8b67b396324c5b2672 -s $BLENDEDMVS_PATH/5b6eff8b67b396324c5b2672 --eval --stereo_depth_sup --lambda_stereo_depth_sup 0.01 --iteration 20000 --iter_stereo_depth_sup 17000 --resolution 1 --baseline_low 2 --baseline_high 4 --port [PORT]
python render.py -m output_blendedMVS/5b6eff8b67b396324c5b2672
python metrics.py -m output_blendedMVS/5b6eff8b67b396324c5b2672
python evaluate_depth.py -m output_blendedMVS/5b6eff8b67b396324c5b2672 --dataset blended

#############################################

python train.py -m output_blendedMVS/5bf18642c50e6f7f8bdbd492 -s $BLENDEDMVS_PATH/5bf18642c50e6f7f8bdbd492 --eval --stereo_depth_sup --lambda_stereo_depth_sup 0.01 --iteration 20000 --iter_stereo_depth_sup 17000 --resolution 1 --baseline_low 2 --baseline_high 4 --port [PORT]
python render.py -m output_blendedMVS/5bf18642c50e6f7f8bdbd492
python metrics.py -m output_blendedMVS/5bf18642c50e6f7f8bdbd492
python evaluate_depth.py -m output_blendedMVS/5bf18642c50e6f7f8bdbd492 --dataset blended

#############################################

python train.py -m output_blendedMVS/5bff3c5cfe0ea555e6bcbf3a -s $BLENDEDMVS_PATH/5bff3c5cfe0ea555e6bcbf3a --eval --stereo_depth_sup --lambda_stereo_depth_sup 0.01 --iteration 20000 --iter_stereo_depth_sup 17000 --resolution 1 --baseline_low 2 --baseline_high 4 --port [PORT]
python render.py -m output_blendedMVS/5bff3c5cfe0ea555e6bcbf3a
python metrics.py -m output_blendedMVS/5bff3c5cfe0ea555e6bcbf3a
python evaluate_depth.py -m output_blendedMVS/5bff3c5cfe0ea555e6bcbf3a --dataset blended
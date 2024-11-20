SCANNETPP_PATH=[PATH_TO_SCANNETPP_DATASET]

python train.py -m output_scannetpp/b20a261fdf -s $SCANNETPP_PATH/b20a261fdf --eval --resolution -1 --stereo_depth_sup --lambda_stereo_depth_sup 0.1 --iter_stereo_depth_sup 7000 --iterations 11000 --baseline_low 0.05 --baseline_high 0.1 --data_device cpu --port [PORT]
python render.py -m output_scannetpp/b20a261fdf
python metrics.py -m output_scannetpp/b20a261fdf
python evaluate_depth.py -m output_scannetpp/b20a261fdf --dataset scannetpp

#############################################

python train.py -m output_scannetpp/8b5caf3398 -s $SCANNETPP_PATH/8b5caf3398 --eval --resolution -1 --stereo_depth_sup --lambda_stereo_depth_sup 0.1 --iter_stereo_depth_sup 7000 --iterations 11000 --baseline_low 0.05 --baseline_high 0.1 --data_device cpu --port [PORT]
python render.py -m output_scannetpp/8b5caf3398
python metrics.py -m output_scannetpp/8b5caf3398
python evaluate_depth.py -m output_scannetpp/8b5caf3398 --dataset scannetpp

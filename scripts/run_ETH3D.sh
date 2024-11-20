ETH_PATH=[PATH_TO_ETH3D_DATASET]

python train.py -m output_ETH3D/courtyard -s $ETH_PATH/courtyard --eval --resolution -1 --stereo_depth_sup --lambda_stereo_depth_sup 0.1 --iter_stereo_depth_sup 7000 --iterations 11000 --baseline_low 0.2 --baseline_high 0.4 --port [PORT]
python render.py -m output_ETH3D/courtyard
python metrics.py -m output_ETH3D/courtyard
python evaluate_depth.py -m output_ETH3D/courtyard

#############################################

python train.py -m output_ETH3D/delivery_area -s $ETH_PATH/delivery_area --eval --resolution -1 --stereo_depth_sup --lambda_stereo_depth_sup 0.1 --iter_stereo_depth_sup 7000 --iterations 11000 --baseline_low 0.2 --baseline_high 0.4 --port [PORT]
python render.py -m output_ETH3D/delivery_area
python metrics.py -m output_ETH3D/delivery_area
python evaluate_depth.py -m output_ETH3D/delivery_area

#############################################

python train.py -m output_ETH3D/electro -s $ETH_PATH/electro --eval --resolution -1 --stereo_depth_sup --lambda_stereo_depth_sup 0.1 --iter_stereo_depth_sup 7000 --iterations 11000 --baseline_low 0.2 --baseline_high 0.4 --port [PORT]
python render.py -m output_ETH3D/electro
python metrics.py -m output_ETH3D/electro
python evaluate_depth.py -m output_ETH3D/electro

#############################################

python train.py -m output_ETH3D/facade -s $ETH_PATH/facade --eval --resolution -1 --stereo_depth_sup --lambda_stereo_depth_sup 0.1 --iter_stereo_depth_sup 7000 --iterations 11000 --baseline_low 0.2 --baseline_high 0.4 --port [PORT]
python render.py -m output_ETH3D/facade
python metrics.py -m output_ETH3D/facade
python evaluate_depth.py -m output_ETH3D/facade

#############################################

python train.py -m output_ETH3D/kicker -s $ETH_PATH/kicker --eval --resolution -1 --stereo_depth_sup --lambda_stereo_depth_sup 0.1 --iter_stereo_depth_sup 7000 --iterations 11000 --baseline_low 0.05 --baseline_high 0.1 --port [PORT]
python render.py -m output_ETH3D/kicker
python metrics.py -m output_ETH3D/kicker
python evaluate_depth.py -m output_ETH3D/kicker

#############################################

python train.py -m output_ETH3D/meadow -s $ETH_PATH/meadow --eval --resolution -1 --stereo_depth_sup --lambda_stereo_depth_sup 0.1 --iter_stereo_depth_sup 7000 --iterations 11000 --baseline_low 0.2 --baseline_high 0.4 --port [PORT]
python render.py -m output_ETH3D/meadow
python metrics.py -m output_ETH3D/meadow
python evaluate_depth.py -m output_ETH3D/meadow

#############################################

python train.py -m output_ETH3D/office -s $ETH_PATH/office --eval --resolution -1 --stereo_depth_sup --lambda_stereo_depth_sup 0.1 --iter_stereo_depth_sup 7000 --iterations 11000 --baseline_low 0.2 --baseline_high 0.4 --port [PORT]
python render.py -m output_ETH3D/office
python metrics.py -m output_ETH3D/office
python evaluate_depth.py -m output_ETH3D/office

#############################################

python train.py -m output_ETH3D/pipes -s $ETH_PATH/pipes --eval --resolution -1 --stereo_depth_sup --lambda_stereo_depth_sup 0.1 --iter_stereo_depth_sup 7000 --iterations 11000 --baseline_low 0.05 --baseline_high 0.1 --port [PORT]
python render.py -m output_ETH3D/pipes
python metrics.py -m output_ETH3D/pipes
python evaluate_depth.py -m output_ETH3D/pipes

#############################################

python train.py -m output_ETH3D/playground -s $ETH_PATH/playground --eval --resolution -1 --stereo_depth_sup --lambda_stereo_depth_sup 0.1 --iter_stereo_depth_sup 7000 --iterations 11000 --baseline_low 0.05 --baseline_high 0.1 --port [PORT]
python render.py -m output_ETH3D/playground
python metrics.py -m output_ETH3D/playground
python evaluate_depth.py -m output_ETH3D/playground

#############################################

python train.py -m output_ETH3D/relief -s $ETH_PATH/relief --eval --resolution -1 --stereo_depth_sup --lambda_stereo_depth_sup 0.1 --iter_stereo_depth_sup 7000 --iterations 11000 --baseline_low 0.2 --baseline_high 0.4 --port [PORT]
python render.py -m output_ETH3D/relief
python metrics.py -m output_ETH3D/relief
python evaluate_depth.py -m output_ETH3D/relief

#############################################

python train.py -m output_ETH3D/relief_2 -s $ETH_PATH/relief_2 --eval --resolution -1 --stereo_depth_sup --lambda_stereo_depth_sup 0.1 --iter_stereo_depth_sup 7000 --iterations 11000 --baseline_low 0.2 --baseline_high 0.4 --port [PORT]
python render.py -m output_ETH3D/relief_2
python metrics.py -m output_ETH3D/relief_2
python evaluate_depth.py -m output_ETH3D/relief_2

#############################################

python train.py -m output_ETH3D/terrace -s $ETH_PATH/terrace --eval --resolution -1 --stereo_depth_sup --lambda_stereo_depth_sup 0.1 --iter_stereo_depth_sup 7000 --iterations 11000 --baseline_low 0.8 --baseline_high 1.2 --port [PORT]
python render.py -m output_ETH3D/terrace
python metrics.py -m output_ETH3D/terrace
python evaluate_depth.py -m output_ETH3D/terrace

#############################################

python train.py -m output_ETH3D/terrains -s $ETH_PATH/terrins --eval --resolution -1 --stereo_depth_sup --lambda_stereo_depth_sup 0.1 --iter_stereo_depth_sup 7000 --iterations 11000 --baseline_low 0.2 --baseline_high 0.4 --port [PORT]
python render.py -m output_ETH3D/terrains
python metrics.py -m output_ETH3D/terrains
python evaluate_depth.py -m output_ETH3D/terrains

#############################################


python train_bulk.py --dataloader=basic_loader --data_input_path=./test/BSDS100_LR/4 --data_truth_path=./test/BSDS100 --model=ifgsm --model_alpha=0.00392157 --model_eps=0.00392157 --sr_config_path=targets/configs/rcan.json --scale=4 --max_steps=50 --train_path=./srattack/rcan/bsd100/eps1 --cuda_device=0

python train_bulk.py --dataloader=basic_loader --data_input_path=./test/BSDS100_LR/4 --data_truth_path=./test/BSDS100 --model=ifgsm --model_alpha=0.00784314 --model_eps=0.00784314 --sr_config_path=targets/configs/rcan.json --scale=4 --max_steps=50 --train_path=./srattack/rcan/bsd100/eps2 --cuda_device=0

python train_bulk.py --dataloader=basic_loader --data_input_path=./test/BSDS100_LR/4 --data_truth_path=./test/BSDS100 --model=ifgsm --model_alpha=0.01568627 --model_eps=0.01568627 --sr_config_path=targets/configs/rcan.json --scale=4 --max_steps=50 --train_path=./srattack/rcan/bsd100/eps4 --cuda_device=0

python train_bulk.py --dataloader=basic_loader --data_input_path=./test/BSDS100_LR/4 --data_truth_path=./test/BSDS100 --model=ifgsm --model_alpha=0.03137255 --model_eps=0.03137255 --sr_config_path=targets/configs/rcan.json --scale=4 --max_steps=50 --train_path=./srattack/rcan/bsd100/eps8 --cuda_device=0

python train_bulk.py --dataloader=basic_loader --data_input_path=./test/BSDS100_LR/4 --data_truth_path=./test/BSDS100 --model=ifgsm --model_alpha=0.0627451 --model_eps=0.0627451 --sr_config_path=targets/configs/rcan.json --scale=4 --max_steps=50 --train_path=./srattack/rcan/bsd100/eps16 --cuda_device=0

python train_bulk.py --dataloader=basic_loader --data_input_path=./test/BSDS100_LR/4 --data_truth_path=./test/BSDS100 --model=ifgsm --model_alpha=0.1254902 --model_eps=0.1254902 --sr_config_path=targets/configs/rcan.json --scale=4 --max_steps=50 --train_path=./srattack/rcan/bsd100/eps32 --cuda_device=0


python train_bulk.py --dataloader=basic_loader --data_input_path=./test/BSDS100_LR/4 --data_truth_path=./test/BSDS100 --model=ifgsm --model_alpha=0.00392157 --model_eps=0.00392157 --sr_config_path=targets/configs/edsr_baseline.json --scale=4 --max_steps=50 --train_path=./srattack/edsr_baseline/bsd100/eps1 --cuda_device=0

python train_bulk.py --dataloader=basic_loader --data_input_path=./test/BSDS100_LR/4 --data_truth_path=./test/BSDS100 --model=ifgsm --model_alpha=0.00784314 --model_eps=0.00784314 --sr_config_path=targets/configs/edsr_baseline.json --scale=4 --max_steps=50 --train_path=./srattack/edsr_baseline/bsd100/eps2 --cuda_device=0

python train_bulk.py --dataloader=basic_loader --data_input_path=./test/BSDS100_LR/4 --data_truth_path=./test/BSDS100 --model=ifgsm --model_alpha=0.01568627 --model_eps=0.01568627 --sr_config_path=targets/configs/edsr_baseline.json --scale=4 --max_steps=50 --train_path=./srattack/edsr_baseline/bsd100/eps4 --cuda_device=0

python train_bulk.py --dataloader=basic_loader --data_input_path=./test/BSDS100_LR/4 --data_truth_path=./test/BSDS100 --model=ifgsm --model_alpha=0.03137255 --model_eps=0.03137255 --sr_config_path=targets/configs/edsr_baseline.json --scale=4 --max_steps=50 --train_path=./srattack/edsr_baseline/bsd100/eps8 --cuda_device=0

python train_bulk.py --dataloader=basic_loader --data_input_path=./test/BSDS100_LR/4 --data_truth_path=./test/BSDS100 --model=ifgsm --model_alpha=0.0627451 --model_eps=0.0627451 --sr_config_path=targets/configs/edsr_baseline.json --scale=4 --max_steps=50 --train_path=./srattack/edsr_baseline/bsd100/eps16 --cuda_device=0

python train_bulk.py --dataloader=basic_loader --data_input_path=./test/BSDS100_LR/4 --data_truth_path=./test/BSDS100 --model=ifgsm --model_alpha=0.1254902 --model_eps=0.1254902 --sr_config_path=targets/configs/edsr_baseline.json --scale=4 --max_steps=50 --train_path=./srattack/edsr_baseline/bsd100/eps32 --cuda_device=0
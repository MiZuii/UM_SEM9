source .venv/bin/activate
python ./baseline_models/trainer.py --dataset HardMNIST --model resnet18 --epoch 8
python ./baseline_models/trainer.py --dataset CIFAR10 --model resnet18
python ./baseline_models/trainer.py --dataset CIFAR100 --model resnet18 --epoch 15
python ./baseline_models/trainer.py --dataset HardMNIST --model resnet34 --epoch 12
python ./baseline_models/trainer.py --dataset CIFAR10 --model resnet34 --epoch 12
python ./baseline_models/trainer.py --dataset CIFAR100 --model resnet34 --epoch 15
python ./baseline_models/trainer.py --dataset HardMNIST --model resnet50 --epoch 15
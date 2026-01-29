source .venv/bin/activate
python ./models_trainers/baseline_trainer.py --dataset HardMNIST --model resnet18 --epoch 8
python ./models_trainers/baseline_trainer.py --dataset CIFAR10 --model resnet18
python ./models_trainers/baseline_trainer.py --dataset CIFAR100 --model resnet18 --epoch 15
python ./models_trainers/baseline_trainer.py --dataset HardMNIST --model resnet34 --epoch 12
python ./models_trainers/baseline_trainer.py --dataset CIFAR10 --model resnet34 --epoch 12
python ./models_trainers/baseline_trainer.py --dataset CIFAR100 --model resnet34 --epoch 15

python ./models_trainers/diet_trainer.py --dataset HardMNIST --model resnet18
python ./models_trainers/diet_trainer.py --dataset CIFAR10 --model resnet18
python ./models_trainers/diet_trainer.py --dataset CIFAR100 --model resnet18
python ./models_trainers/diet_trainer.py --dataset HardMNIST --model resnet34
python ./models_trainers/diet_trainer.py --dataset CIFAR10 --model resnet34
python ./models_trainers/diet_trainer.py --dataset CIFAR100 --model resnet34
import subprocess

command = "CUDA_VISIBLE_DEVICES=0 python main.py --lr 0.0005 --epoch 1000 --dataset movie --N 16 --w 1"
print('Running', command)
subprocess.call(command, shell=True)

command = "CUDA_VISIBLE_DEVICES=0 python main.py --lr 0.001 --epoch 1000 --dataset gowalla --N 8 --w 1"
print('Running', command)
subprocess.call(command, shell=True)


command = "CUDA_VISIBLE_DEVICES=0 python main.py --lr 0.001 --epoch 1000 --dataset pinterest --N 8 --w 1"
print('Running', command)
subprocess.call(command, shell=True)

command = "CUDA_VISIBLE_DEVICES=0 python main.py --lr 0.001 --epoch 1000 --dataset yelp --N 8 --w 1"
print('Running', command)
subprocess.call(command, shell=True)


command = "CUDA_VISIBLE_DEVICES=0 python main.py --lr 0.001 --epoch 1000 --dataset book --N 2 --w 1"
print('Running', command)
subprocess.call(command, shell=True)

command = "CUDA_VISIBLE_DEVICES=0 python main.py --lr 0.0001 --epoch 20 --dataset dianping --N 4 --w 0.5"
print('Running', command)
subprocess.call(command, shell=True)
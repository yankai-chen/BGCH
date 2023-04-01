# BGCH

This is the PyTorch implementation for our WWW2023 paper:
"Bipartite Graph Convolutional Hashing for Effective and Efficient Top-N Search in Hamming Space."
*Chen, Yankai, Yixiang Fang, Yifei Zhang and Irwin King.* WWW'23.
It is currently available in [Arxiv](https://arxiv.org/abs/xxx).


## Environment Requirement

The code runs well under python 3.8. The required packages are referred to <b>env.txt</b>.

## Datasets

All datasets except are directly available in the folder "BGCH/Dataset/*". For the largest dataset "Dianping", please refer to [Link](https://drive.google.com/file/d/1FOmx6-8fYd2vkg2CFA0kx5zNShpdmbRY/view?usp=sharing).

## To Run the codes under "BGCH/src/*"

<li> <b>Simply specify the dataset as well as the settings as follows</b>:
```

python main.py --dataset xxx ...

``
For the suggested settings for all datasets, please refer to the file <b>run_all.py</b>



## Citation
Please kindly cite our paper if you find our codes useful for your research and work:

```
@inproceedings{bgch,
  author={Chen, Yankai and Fang, Yixiang and Zhang, Yifei and King, Irwin},
  title     = {Bipartite Graph Convolutional Hashing for Effective and Efficient Top-N Search in Hamming Space},
  booktitle = {{WWW} '23: The ACM Web Conference, AUSTIN, TEXAS, USA, APRIL 30 - MAY 4, 2023},
  publisher = {{ACM}},
  year      = {2023}
}

```

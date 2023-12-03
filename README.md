# Speaker separation project

This repository contains vocoder implementation done as a part of homework #4 for the DLA course at the CS Faculty of HSE. See [wandb report](https://wandb.ai/ira_gl/dla_vocoder/reports/Vocoder-report--Vmlldzo2MTU0MjMy?accessToken=b50t3dmb60tzlrqthfber0zh7nm2w0xkp0ehcuokh1o82d861m3ns6zk5bs17cjc) 

## Installation guide

Clone this repository. Move to corresponding folder and install required packages:

```shell
git clone https://github.com/ir4kgl/vocoder_dla
cd vocoder_dla
pip install -r ./requirements.txt
```

## Checkpoint

To download the final checkpoint run 

```shell
python3 download_checkpoint.py
```

it will download final checkpoint in `checkpoints/final_run` folder.

## Run train

To train model simply run

```shell
python3 train.py --c config.json --r CHECKPOINT.pth
```

where `config.json` is configuration file with all data, model and trainer parameters and `CHECKPOINT.pth` is an optional argument to continue training starting from a given checkpoint. 

Configuration of my final experiment you can find in the file `configs/final_run_config.json`.




## HiFi-GAN Model

File `vocoder/model/hifigan.py` contains the implementation of [](https://arxiv.org/abs/2010.05646) model. The code was written by me diligently studying the original article, however I took some inspiration and implementation of auxiliary methods (like `get_padding`) from  [this HiFi-GAN implementation](https://github.com/jik876/hifi-gan).


## Loss function and metrics

See implementation of the loss function in `vocoder/loss/` module. Here you can find implementation of adversarial loss for generator and discriminator, additional feature map loss for generator and mel loss for generator in files `AdvLoss.py`, `FMLoss.py` and `MelLoss.py` respectively.
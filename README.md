# Starting

First, git clone this repo and checkout to `dev` branch

```bash
git clone https://github.com/FastMRI-BrainMonster/challenge.git
cd challenge
git checkout dev
```

Install wandb and disable it (we used it for our parameter searching!)

```bash
pip install wandb
export WANDB_MODE="disabled"
```

Install pygrappa for generating GRAPPA files (better than given ones)

```bash
pip install pygrappa
```

Now, you should generate the brain mask files and grappa files. From our ‘challenge’ folder, 

```bash
pip install opencv-python
python utils/data/preprocessing.py
```

If you get an AttributeError (module ‘numpy’ has no attribute ‘typeDict’), downgrade your numpy version to 1.21, like this:

```bash
pip install numpy==1.21
```

This will make a `brain_mask` folder. For grappa files, run these:

```bash
python utils/data/generate_grappa_acc4_train.py
python utils/data/generate_grappa_acc4_val.py
python utils/data/generate_grappa_acc4_leaderboard.py
python utils/data/generate_grappa_acc8_train.py
python utils/data/generate_grappa_acc8_val.py
python utils/data/generate_grappa_acc8_leaderboard.py
```

You are ready now! Train the E2E-VarNet by

```bash
python train.py --num-epochs 100 --cascade 6 --chans 12 --sens_chans 5
```

Because of the MRAugmentation strength, the total “num epochs” are important and also very meaningful for the result. Our best model (SSIM 0.982, just using E2E-VarNet) was found at epoch 61 with cascade 6, chans 12 and sens_chans 5, when total 100 epochs were given as the input hyperparameter. 

After training, reconstruct the images (using your best E2E-VarNet). In this code, we assume that our `best_model.pt` file is under the “../result/test_varnet/checkpoints” folder.

```bash
python save_images.py --cascade 6 --chans 12 --sens_chans 5
python save_images_l.py --cascade 6 --chans 12 --sens_chans 5
```

[WIP] UNET
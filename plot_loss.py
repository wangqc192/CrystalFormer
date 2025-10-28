import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('csv_path')
args = parser.parse_args()
df = pd.read_csv(args.csv_path, sep=r'\s+')
tr_loss_w = df["t_loss_w"]
tr_loss_a = df["t_loss_a"]
tr_loss_x = df["t_loss_xyz"]
tr_loss_l = df["t_loss_l"]

va_loss_w = df["v_loss_w"]
va_loss_a = df["v_loss_a"]
va_loss_x = df["v_loss_xyz"]
va_loss_l = df["v_loss_l"]

epoch = df["epoch"]
train_epoch = epoch
val_epoch = epoch

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs[0, 0].plot(train_epoch, tr_loss_w, label='train', color='red')
axs[0, 0].plot(val_epoch, va_loss_w, label='val', color='gray')

axs[0, 0].set_title('W loss')
axs[0, 0].set_xlabel('epochs')
axs[0, 0].set_ylabel('loss')
axs[0, 0].legend()


axs[0, 1].plot(train_epoch, tr_loss_a, label='train', color='red')
axs[0, 1].plot(val_epoch, va_loss_a, label='val', color='gray')

axs[0, 1].set_title('A loss')
axs[0, 1].set_xlabel('epochs')
axs[0, 1].set_ylabel('loss')
axs[0, 1].legend()


axs[1, 0].plot(train_epoch, tr_loss_x, label='train', color='red')
axs[1, 0].plot(val_epoch, va_loss_x, label='val', color='gray')

axs[1, 0].set_title('X loss')
axs[1, 0].set_xlabel('epochs')
axs[1, 0].set_ylabel('loss')
axs[1, 0].legend()



axs[1, 1].plot(train_epoch, tr_loss_l, label='train', color='red')
axs[1, 1].plot(val_epoch, va_loss_l, label='val', color='gray')

axs[1, 1].set_title('L loss')
axs[1, 1].set_xlabel('epochs')
axs[1, 1].set_ylabel('loss')
axs[1, 1].legend()

save_path = os.path.dirname(args.csv_path) + 'loss_curves.png'
plt.savefig(save_path, dpi=300, bbox_inches="tight")
print("\n========== Calculate the loss of test dataset ==========")
import picle
import math
import numpy as np
import jax
import crystalformer.src.checkpoint as checkpoint

from crystalformer.src.transformer import make_transformer
from crystalformer.src.loss import make_loss_fn

key = jax.random.PRNGKey(42)
data_path = ''
ckpt_file = ''

np.set_printoptions(threshold=np.inf)
test_data = picle(open(data_path, "r"))
batchsize =128

params, transformer = make_transformer(key, 5, 16, 4, 21, 
                                      256, 16, 16, 64, 64, 32, 
                                      119, 28, 0.5)
ckpt = checkpoint.load_data(ckpt_file)
params = ckpt["params"]

loss_fn, logp_fn = make_loss_fn(21, 119, 28, 16, 4, transformer, 1, 1, 1)
test_G, test_L, test_XYZ, test_A, test_W = test_data
print (test_G.shape, test_L.shape, test_XYZ.shape, test_A.shape, test_W.shape)
test_loss = 0
num_samples = len(test_L)
num_batches = math.ceil(num_samples / batchsize)
for batch_idx in range(num_batches):
    start_idx = batch_idx * batchsize
    end_idx = min(start_idx + batchsize, num_samples)
    G, L, XYZ, A, W = test_G[start_idx:end_idx], \
                        test_L[start_idx:end_idx], \
                        test_XYZ[start_idx:end_idx], \
                        test_A[start_idx:end_idx], \
                        test_W[start_idx:end_idx]
    loss, _ = jax.jit(loss_fn, static_argnums=7)(params, key, G, L, XYZ, A, W, False)
    test_loss += loss
test_loss = test_loss / num_batches

from bm3d import bm3d
import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.data import cat
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import time

from bm3d_cupy import bm3d_gpu
from bm3d_torch import bm3d_torch


rng = np.random.default_rng(seed=0)


x = cat().astype('float32') / 255.0
sigma = 0.1
y = (x + rng.standard_normal(x.shape).astype('float32') * sigma).astype('float32')


def benchmark_cpu(y, sigma, n_runs=10):
    bm3d(y, sigma)
    times = []
    out = None
    for _ in range(n_runs):
        tic = time.perf_counter()
        out = bm3d(y, sigma)
        times.append(time.perf_counter() - tic)
    return out, np.asarray(times, dtype=np.float64)


def benchmark_gpu(y_gpu, sigma, n_runs=10, chunk_size=2048):
    bm3d_gpu(y_gpu, sigma, chunk_size=chunk_size)
    cp.cuda.Stream.null.synchronize()
    times = []
    out = None
    for _ in range(n_runs):
        cp.cuda.Stream.null.synchronize()
        tic = time.perf_counter()
        out = bm3d_gpu(y_gpu, sigma, chunk_size=chunk_size)
        cp.cuda.Stream.null.synchronize()
        times.append(time.perf_counter() - tic)
    return out, np.asarray(times, dtype=np.float64)


def benchmark_torch(y_torch, sigma, n_runs=10, chunk_size=2048):
    bm3d_torch(y_torch, sigma, chunk_size=chunk_size)
    if y_torch.is_cuda:
        torch.cuda.synchronize()
    times = []
    out = None
    for _ in range(n_runs):
        if y_torch.is_cuda:
            torch.cuda.synchronize()
        tic = time.perf_counter()
        out = bm3d_torch(y_torch, sigma, chunk_size=chunk_size)
        if y_torch.is_cuda:
            torch.cuda.synchronize()
        times.append(time.perf_counter() - tic)
    return out, np.asarray(times, dtype=np.float64)


z, cpu_times = benchmark_cpu(y, sigma, n_runs=10)
psnr_val = psnr(x, z, data_range=1.0)
ssim_val = ssim(x, z, data_range=1.0, channel_axis=-1)
print(f'BM3D (CPU) time: {cpu_times.mean():.4f} ({cpu_times.std():.4f}) s')
print(f'  PSNR: {psnr_val:.2f} dB')
print(f'  SSIM: {ssim_val:.4f}')
print('\n')

x_gpu = cp.asarray(x)
y_gpu = cp.asarray(y)
z_gpu, gpu_times = benchmark_gpu(y_gpu, sigma, n_runs=10, chunk_size=2048)
psnr_val_gpu = psnr(x_gpu.get(), z_gpu.get(), data_range=1.0)
ssim_val_gpu = ssim(x_gpu.get(), z_gpu.get(), data_range=1.0, channel_axis=-1)
speedup = cpu_times.mean() / gpu_times.mean()
print(f'BM3D (CuPy) time: {gpu_times.mean():.4f} ({gpu_times.std():.4f}) s')
print(f'  Speedup: {speedup:.2f}x')
print(f'  PSNR: {psnr_val_gpu:.2f} dB')
print(f'  SSIM: {ssim_val_gpu:.4f}')
print('\n')

device = torch.device('cuda')
x_torch = torch.from_numpy(x).to(device=device)
y_torch = torch.from_numpy(y).to(device=device)
z_torch, torch_times = benchmark_torch(y_torch, sigma, n_runs=10, chunk_size=2048)
z_torch_np = z_torch.detach().cpu().numpy()
psnr_val_torch = psnr(x, z_torch_np, data_range=1.0)
ssim_val_torch = ssim(x, z_torch_np, data_range=1.0, channel_axis=-1)
speedup_torch = cpu_times.mean() / torch_times.mean()
device_name = device.type.upper()
print(f'BM3D (Torch) time: {torch_times.mean():.4f} ({torch_times.std():.4f}) s')
print(f'  Speedup: {speedup_torch:.2f}x')
print(f'  PSNR: {psnr_val_torch:.2f} dB')
print(f'  SSIM: {ssim_val_torch:.4f}')
print('\n')

fig, axes = plt.subplots(1, 5, figsize=(16.0, 2.4), layout='constrained')
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False); ax.spines['left'].set_visible(False)
axes[0].imshow(np.clip(x, 0, 1))
axes[0].set_title('Clean')
axes[1].imshow(np.clip(y, 0, 1))
axes[1].set_title('Noisy')
axes[2].imshow(np.clip(z, 0, 1))
axes[2].set_title(f'BM3D (CPU)\nPSNR: {psnr_val:.2f} dB, SSIM: {ssim_val:.4f}')
axes[3].imshow(np.clip(z_gpu.get(), 0, 1))
axes[3].set_title(f'BM3D (CuPy)\nPSNR: {psnr_val_gpu:.2f} dB, SSIM: {ssim_val_gpu:.4f}')
axes[4].imshow(np.clip(z_torch_np, 0, 1))
axes[4].set_title(f'BM3D (Torch)\nPSNR: {psnr_val_torch:.2f} dB, SSIM: {ssim_val_torch:.4f}')
fig.savefig('bm3d.png', dpi=300, bbox_inches='tight')
plt.show()

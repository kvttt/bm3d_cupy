"""GPU-accelerated BM3D denoising built on CuPy."""

try:
    import cupy as cp
    import pywt
    from cupyx.scipy.fft import dct
except ImportError as exc:
    raise ImportError(
        "bm3d_cupy requires CuPy and PyWavelets. Install a matching CuPy wheel, "
        "for example 'pip install \"bm3d-cupy[cuda13]\"', before importing this module."
    ) from exc

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="pywt")

__version__ = "0.1.2"
__all__ = [
    "bm3d_gpu",
    "get_dct_matrix",
    "get_transform_matrices",
    "get_wavelet_matrix",
]


EPS = 2 ** (-52)


def get_wavelet_matrix(n, wavelet, level=None):
    if level is None:
        level = n.bit_length() - 1
    I = cp.eye(n)
    matrix = cp.empty((n, n), dtype=I.dtype)
    for i in range(n):
        basis = cp.asnumpy(I[:, i])
        coeffs = pywt.wavedec(basis, wavelet, mode='periodization', level=level)
        matrix[:, i] = cp.hstack([cp.asarray(coeff) for coeff in coeffs])
    matrix = matrix / cp.linalg.norm(matrix, axis=0)
    return matrix, cp.linalg.inv(matrix)


def get_dct_matrix(n, norm='ortho'):
    matrix = dct(cp.eye(n), norm=norm, axis=0)
    inverse = matrix.T if norm == 'ortho' else cp.linalg.inv(matrix)
    return matrix, inverse


def get_transform_matrices(n, transform_name):
    n = int(n)
    transform_name = transform_name.lower()
    if transform_name == 'dct':
        return get_dct_matrix(n, norm='ortho')
    elif (n & (n - 1)) == 0:
        return get_wavelet_matrix(n, transform_name)
    raise ValueError(f"Unsupported transform '{transform_name}' for size n={n}.")


def bm3d_gpu(
    y,
    sigma,
    patch_size=8,
    ht_group_size=16,
    wiener_group_size=32,
    search_radius=19,
    search_step=1,
    ref_stride=3,
    hard_threshold=3.0,
    wiener_mu2=0.4,
    chunk_size=2048,
):
    # y
    y = cp.asarray(y, dtype=cp.float32)
    squeeze = y.ndim == 2
    if squeeze:
        y = y[..., cp.newaxis]  # (h, w, c)
    h, w, c = y.shape
    # sigma
    sigma = cp.asarray(sigma, dtype=cp.float32)
    if sigma.ndim == 0:
        sigma = cp.full(c, float(sigma), dtype=cp.float32)  # (c,)
    elif sigma.size == c:
        sigma = sigma.reshape(c).astype(cp.float32, copy=False)  # (c,)
    if cp.all(sigma <= 0):  # edge case: no denoising
        return y[..., 0] if squeeze else y
    sigma_ch = cp.asarray(sigma, dtype=cp.float32)  # (c,)
    sigma2_ch = sigma_ch * sigma_ch  # (c,)
    # patch_size
    p = min(int(patch_size), h, w) 
    # grouping settings
    search_radius = max(0, int(search_radius))
    search_step = max(1, int(search_step))
    ref_stride = max(1, int(ref_stride))
    chunk_size = max(1, int(chunk_size))  # number of groups to process in parallel
    offset_radius = search_radius // search_step  # search radius (steps)
    hp, wp = h - p + 1, w - p + 1  # hp*wp = total number of patches
    n_candidates = (2 * offset_radius + 1) ** 2  # upper bound on group size
    ht_group_size = int(min(max(1, int(ht_group_size)), n_candidates))  # stage 1 (hard-threshold) group size
    wiener_group_size = int(min(max(1, int(wiener_group_size)), n_candidates))  # stage 2 (Wiener filtering) group size
    # transform matrices
    def _transform_matrices(n, transform_name):  # get transform matrices
        forward, inverse = get_transform_matrices(n, transform_name)
        return cp.asarray(forward, dtype=cp.float32), cp.asarray(inverse, dtype=cp.float32)
    # 2d transforms (along spatial dimensions)
    spatial_ht, spatial_ht_inv = _transform_matrices(p, 'bior1.5')  # biorthogonal WT, (p, p)
    spatial_wiener, spatial_wiener_inv = _transform_matrices(p, 'dct')  # DCT, (p, p)
    # 1d transforms (along group dimension)
    group_ht, group_ht_inv = _transform_matrices(ht_group_size, 'haar')  # Haar, (k_ht, k_ht)
    group_wiener, group_wiener_inv = _transform_matrices(wiener_group_size, 'haar')  # Haar, (k_w, k_w)
    # Kaiser window, applied to weights during aggregation to reduce boundary artifact
    win_1d = cp.kaiser(p, 2.0).astype(cp.float32)  # (p,)
    win = win_1d[:, None] * win_1d[None, :]  # (p, p)
    # helper functions
    def _patch_tensor(img):  # extract all `hp*wp` patches
        patches = cp.lib.stride_tricks.sliding_window_view(img, (p, p), axis=(0, 1))  # Result: (hp, wp, c, p, p)
        return cp.ascontiguousarray(patches.reshape(-1, img.shape[2], p, p))  # (hp*wp, c, p, p)
    def _reference_grid():  # get locations of all reference patches
        ys = cp.arange(0, hp, ref_stride, dtype=cp.int32)
        xs = cp.arange(0, wp, ref_stride, dtype=cp.int32)
        # always include the last patch to avoid boundary artifact
        if ((hp - 1) // ref_stride) * ref_stride != hp - 1:
            ys = cp.concatenate((ys, cp.asarray([hp - 1], dtype=cp.int32)))
        if ((wp - 1) // ref_stride) * ref_stride != wp - 1:
            xs = cp.concatenate((xs, cp.asarray([wp - 1], dtype=cp.int32)))
        yy, xx = cp.meshgrid(ys, xs, indexing='ij')
        return yy.ravel(), xx.ravel()  # (n_refs,), (n_refs,)
    def _search_offsets():  # get candidate offsets
        offsets = cp.arange(-offset_radius * search_step, offset_radius * search_step + 1, search_step, dtype=cp.int32)  # (2*offset_radius + 1,)
        yy, xx = cp.meshgrid(offsets, offsets, indexing='ij')
        return yy.ravel(), xx.ravel()  # (n_candidates,), (n_candidates,)
    ref_y, ref_x = _reference_grid()  # (n_refs,), (n_refs,)
    off_y, off_x = _search_offsets()  # (n_candidates,), (n_candidates,)
    n_refs = int(ref_y.size)  # total number of reference patches
    # helper arrays
    patch_y = cp.arange(p, dtype=cp.int32)[None, None, None, :, None]  # (1, 1, 1, p, 1)
    patch_x = cp.arange(p, dtype=cp.int32)[None, None, None, None, :]  # (1, 1, 1, 1, p)
    patch_c = cp.arange(c, dtype=cp.int32)[None, None, :, None, None]  # (1, 1, c, 1, 1)
    # functions for transforms
    def _spatial_forward(blocks, transform): 
        tmp = cp.einsum('ai,nkcij->nkcaj', transform, blocks, optimize=True)
        return cp.einsum('bj,nkcaj->nkcab', transform, tmp, optimize=True)
    def _spatial_inverse(coeffs, inverse):
        tmp = cp.einsum('ia,nkcab->nkcib', inverse, coeffs, optimize=True)
        return cp.einsum('jb,nkcib->nkcij', inverse, tmp, optimize=True)
    def _group_forward(blocks, transform):
        return cp.einsum('mk,nkcab->nmcab', transform, blocks, optimize=True)
    def _group_inverse(coeffs, inverse):
        return cp.einsum('km,nmcab->nkcab', inverse, coeffs, optimize=True)
    # functions for aggregation
    def _aggregate(accum, weight, pos_y, pos_x, patches, patch_weight):
        # all `p` pixel indices within all `k` patches for all `n` groups
        iy = pos_y[:, :, None, None, None] + patch_y  # (n, k, 1, p, 1)
        ix = pos_x[:, :, None, None, None] + patch_x  # (n, k, 1, 1, p)
        idx = ((iy * w + ix) * c + patch_c).ravel()
        weighted_window = win[None, None, None, :, :] * patch_weight  # (n, 1, c, p, p)
        values = patches * weighted_window  # (n, k, c, p, p)
        weights = cp.broadcast_to(weighted_window, patches.shape)  # (n, k, c, p, p)
        cp.add.at(accum, idx, values.ravel())  # accum[idx] += values.ravel()
        cp.add.at(weight, idx, weights.ravel())  # weight[idx] += weights.ravel()
    # main routine for both stages
    def _estimate(
        match_patches,
        noisy_patches,
        group_size,
        spatial_transform,
        spatial_inverse,
        group_transform,
        group_inverse,
        stage,
    ):
        # patch features
        match_metric = match_patches[:, 0, :, :].reshape(-1, p * p)  # (hp*wp, p*p)
        patch_norm = cp.float32(p * p)
        # initialize accumulators
        accum = cp.zeros(h * w * c, dtype=cp.float32)  # (h*w*c,)
        weight = cp.zeros_like(accum)  # (h*w*c,)
        # process `chunk_size` groups in parallel
        for start in range(0, n_refs, chunk_size):
            # indices handling
            end = min(start + chunk_size, n_refs)
            ry = ref_y[start:end]  # (chunk_size,)
            rx = ref_x[start:end]  # (chunk_size,)
            ref_idx = ry * wp + rx  # (chunk_size,) 
            cand_y_raw = ry[:, None] + off_y[None, :]  # (chunk_size, n_candidates)
            cand_x_raw = rx[:, None] + off_x[None, :]  # (chunk_size, n_candidates)
            valid = ((cand_y_raw >= 0) & (cand_y_raw < hp) & (cand_x_raw >= 0) & (cand_x_raw < wp))  # (chunk_size, n_candidates)
            cand_y = cp.clip(cand_y_raw, 0, hp - 1)  # (chunk_size, n_candidates)
            cand_x = cp.clip(cand_x_raw, 0, wp - 1)  # (chunk_size, n_candidates)
            cand_idx = cand_y * wp + cand_x  # (chunk_size, n_candidates)
            # block matching
            ref = match_metric[ref_idx]  # (chunk_size, p*p)
            cand = match_metric[cand_idx]  # (chunk_size, n_candidates, p*p)
            dist = cp.sum((cand - ref[:, None, :]) ** 2, axis=2) / patch_norm  # (chunk_size, n_candidates)
            dist = cp.where(valid, dist, cp.inf)  # (chunk_size, n_candidates)
            nearest = cp.argpartition(dist, group_size - 1, axis=1)[:, :group_size]  # select top-`group_size`
            nearest_dist = cp.take_along_axis(dist, nearest, axis=1)  # (chunk_size, group_size)
            order = cp.argsort(nearest_dist, axis=1)  # (chunk_size, group_size)
            nearest = cp.take_along_axis(nearest, order, axis=1)  # (chunk_size, group_size)
            group_y = cp.take_along_axis(cand_y, nearest, axis=1)  # (chunk_size, group_size)
            group_x = cp.take_along_axis(cand_x, nearest, axis=1)  # (chunk_size, group_size)
            group_idx = cp.take_along_axis(cand_idx, nearest, axis=1)  # (chunk_size, group_size)
            # extract matched patches from noisy image and stack into a group
            noisy_group = noisy_patches[group_idx].reshape(-1, group_size, c, p, p)  # (chunk_size, group_size, c, p, p)
            # forward 3d transform
            noisy_coeff = _group_forward(_spatial_forward(noisy_group, spatial_transform), group_transform)  # (chunk_size, group_size, c, p, p)
            # stage 1 (hard-threshold)
            if stage == 'hard':
                threshold = cp.float32(hard_threshold) * sigma_ch[None, None, :, None, None]  # (1, 1, c, 1, 1)
                mask = cp.abs(noisy_coeff) >= threshold  # (chunk_size, group_size, c, p, p)
                coeff = noisy_coeff * mask
                denom = cp.maximum(mask.sum(axis=(1, 3, 4)).astype(cp.float32), cp.float32(1.0))  # (chunk_size, group_size)
            # stage 2 (Wiener filtering)
            else:
                pilot_group = match_patches[group_idx].reshape(-1, group_size, c, p, p)  # (chunk_size, group_size, c, p, p)
                pilot_coeff = _group_forward(_spatial_forward(pilot_group, spatial_transform), group_transform)  # (chunk_size, group_size, c, p, p)
                # compute gain
                pilot_power = pilot_coeff * pilot_coeff  # (chunk_size, group_size, c, p, p)
                noise_power = cp.float32(wiener_mu2) * sigma2_ch[None, None, :, None, None]  # (1, 1, c, 1, 1)
                wiener = pilot_power / (pilot_power + noise_power)  # (chunk_size, group_size, c, p, p)
                # apply gain
                coeff = noisy_coeff * wiener 
                denom = cp.maximum(cp.sum(wiener * wiener, axis=(1, 3, 4)), cp.float32(1.0))  # (chunk_size, group_size)
            # inverse 3d transform
            filtered = _spatial_inverse(_group_inverse(coeff, group_inverse), spatial_inverse)  # (chunk_size, group_size, c, p, p)
            # compute aggregation weights
            patch_weight = (1.0 / (sigma2_ch[None, :] * denom))[:, None, :, None, None]  # (chunk_size, 1, c, 1, 1)
            # aggregate
            _aggregate(accum, weight, group_y, group_x, filtered, patch_weight)
        # normalize
        out = accum.reshape(h, w, c) / cp.maximum(weight.reshape(h, w, c), cp.float32(EPS))  # (h, w, c)
        return out
    # stage 1 (hard-threshold)
    y_patches = _patch_tensor(y)  # (hp*wp, c, p, p)
    basic = _estimate(
        y_patches, 
        y_patches,
        ht_group_size,
        spatial_ht,
        spatial_ht_inv,
        group_ht,
        group_ht_inv,
        'hard',
    )  # (h, w, c)
    # stage 2 (Wiener filtering)
    basic_patches = _patch_tensor(basic)  # (hp*wp, c, p, p)
    denoised = _estimate(
        basic_patches, 
        y_patches, 
        wiener_group_size,
        spatial_wiener,
        spatial_wiener_inv,
        group_wiener,
        group_wiener_inv,
        'wiener',
    )  # (h, w, c)
    return denoised[..., 0] if squeeze else denoised

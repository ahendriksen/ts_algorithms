import tomosipo as ts
import torch
from torch.fft import rfft, irfft
import numpy as np


def total_padded_width(width):
    (pad_l, pad_r) = num_pad(width)
    return pad_l + width + pad_r


def num_pad(width):
    # If the width of the filter equals `width`, then we need at least
    # `width - 1` padding. For performance reasons, we prefer the data
    # we feed into the FFT to have an even length. So we use `width`.
    num_padding = width
    num_pad_left = num_padding // 2
    return (num_pad_left, num_padding - num_pad_left)


def pad(sino):
    (num_slices, num_angles, num_pixels) = sino.shape
    num_pad_left, num_pad_right = num_pad(num_pixels)
    num_padding = num_pad_left + num_pad_right
    M = num_pixels + num_padding

    tmp_sino = sino.new_zeros((num_slices, num_angles, M))
    tmp_sino[:, :, num_pad_left:num_pad_left + num_pixels] = sino
    return tmp_sino


def unpad(sino, width):
    pad_left, pad_right = num_pad(width)
    return sino[..., pad_left:pad_left+width]


def pad_filter(h):
    width = 2 * len(h)
    out = h.new_zeros(width)
    w = len(h)
    out[0] = h[0]
    out[1:w] = h[1:]
    out[-w + 1:] = torch.flip(h[1:], dims=(0,))
    return out


def ram_lak(n):
    # Returns a ram_lak filter in filter space.
    # Complex component equals zero.
    filter = torch.zeros(n)
    filter[0] = 0.25
    # even indices are zero
    # for odd indices j, filter[j] equals
    #   -1 / (pi * j) ** 2,          when 2 * j <= n
    #   -1 / (pi * (n - j)) ** 2,    when 2 * j >  n
    odd_indices = torch.arange(1, n, 2)
    cond = 2 * odd_indices > n
    odd_indices[cond] = n - odd_indices[cond]
    filter[1::2] = -1 / (np.pi * odd_indices) ** 2

    return filter


def filter_sino(y, filter=None, padded=True):
    """Filter sinogram for use in FBP

    :param y: `torch.tensor`
        A three-dimensional tensor in sinogram format (height, num_angles, width).

    :param filter: `torch.tensor` (optional)
        If not specified, the ram-lak filter is used. This should be
        one-dimensional tensor that is as wide as the sinogram `y`.

    :param padded: `bool`
        By default, the reconstruction is zero-padded as it is
        filtered. Padding can be skipped by setting `padded=False`.

    :returns:
        A sinogram filtered with the provided filter.
    :rtype: `torch.tensor`
    """
    # Add padding
    original_width = y.shape[-1]

    if filter is None:
        # Use Ram-Lak filter by default.
        filter = ram_lak(y.shape[-1]).to(y.device)

    if padded:
        y = pad(y)
        if len(filter) == original_width:
            filter = pad_filter(filter)

    if filter.shape != y.shape[-1:]:
        raise ValueError(
            f"Filter is the wrong length. Expected length: {y.shape[-1]}. "
            f"Got: {filter.shape}"
        )

    # Fourier transform of sinogram and filter.
    y_f = rfft(y)
    h_f = rfft(filter)

    # Filter the sinogram using complex multiplication:
    y_f *= h_f

    # Invert fourier transform.
    # Make sure inverted data matches the shape of y (for
    # sinograms with odd width).
    y_filtered = irfft(y_f, n=y.shape[-1])

    # Remove padding
    if padded:
        y_filtered = unpad(y_filtered, original_width)

    return y_filtered


def fbp(A, y, padded=True, filter=None, batch_size=10, overwrite_y=False):
    """Compute FBP reconstruction

    If `y` is located on GPU, the entire algorithm is executed on a single GPU.

    If `y` is located in RAM (CPU in PyTorch parlance), then only the
    foward and backprojection are executed on GPU.

    The algorithm is explained in detail in [1].

    :param A: `tomosipo.operator`
        The tomographic operator.

    :param y: `torch.tensor`
        A three-dimensional tensor in sinogram format (height, num_angles, width).

    :param padded: `bool`
        By default, the reconstruction is zero-padded as it is
        filtered. Padding can be skipped by setting `padded=False`.

    :param filter: `torch.tensor` (optional)
        If not specified, the ram-lak filter is used. This should be
        one-dimensional tensor that is as wide as the sinogram `y`.

    :param batch_size: `int`
        Specifies how many projection images will be filtered at the
        same time. Increasing the batch_size will increase the used
        memory. Computation time can be marginally improved by
        tweaking this parameter.

    :param overwrite_y: `bool`
        Specifies whether to overwrite y with the filtered version
        while running this function. Choose `overwrite_y=False` if you
        still want to use y after calling this function. Choose
        `overwrite_y=True` if you would otherwise run out of memory.

    :returns:
        A reconstruction computed using the FBP algorithm.

    :rtype: `torch.tensor`

    [1] Zeng, G. L., Revisit of the ramp filter, IEEE Transactions on
    Nuclear Science, 62(1), 131–136 (2015).
    http://dx.doi.org/10.1109/tns.2014.2363776

    """

    if overwrite_y:
        y_filtered = y
    else:
        y_filtered = torch.empty_like(y)

    if filter is None:
        filter = ram_lak(y.shape[-1]).to(y.device)

    if padded:
        expected_filter_width = total_padded_width(y.shape[-1])
    else:
        expected_filter_width = y.shape[-1]

    if padded and len(filter) != expected_filter_width:
        filter = pad_filter(filter)

    if filter is not None and filter.shape[-1] != expected_filter_width:
        raise ValueError(
            f"Filter is the wrong length. "
            f"Expected length: {expected_filter_width}. "
            f"Got: {filter.shape}. "
        )

    # Filter the sinogram in batches
    for batch_start in range(0, y.shape[1], batch_size):
        batch_end = min(batch_start + batch_size, y.shape[1])
        batch = y[:, batch_start:batch_end, :]
        batch_filtered = filter_sino(batch, filter=filter, padded=padded)

        y_filtered[:, batch_start:batch_end, :] = batch_filtered

    # Backproject the filtered sinogram to obtain a reconstruction
    rec = A.T(y_filtered)

    # Scale result to make sure that fbp(A, A(x)) == x holds at least
    # to some approximation. In limited experiments, this is true for
    # this version of FBP up to 1%.
    # *Note*: For some reason, we do not have to scale with respect to
    # the pixel dimension that is orthogonal to the rotation axis (`u`
    # or horizontal pixel dimension). Hence, we only scale with the
    # other pixel dimension (`v` or vertical pixel dimension).
    vg, pg = A.astra_compat_vg, A.astra_compat_pg

    pixel_height = (pg.det_size[0] / pg.det_shape[0])
    voxel_volume = np.prod(np.array(vg.size / np.array(vg.shape)))
    scaling = (np.pi / pg.num_angles) * pixel_height / voxel_volume

    rec *= scaling

    return rec

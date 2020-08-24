import tomosipo as ts
import torch
import numpy as np
import math


def num_pad(width):
    # We must have:
    # 1) num_pixels + num_padding is even (because rfft wants the #input elements to be even)
    # 2) num_padding // 2 must equal at least num_pixels (not so sure about this actually..)
    if width % 2 == 0:
        num_padding = width
    else:
        num_padding = width + 2

    # num_padding < num_padding_left
    num_pad_left = num_padding // 2
    return (num_pad_left, num_padding - num_pad_left)


def pad(sino):
    (num_slices, num_angles, num_pixels) = sino.shape
    num_pad_left, num_pad_right = num_pad(num_pixels)
    num_padding = num_pad_left + num_pad_right
    # M is always even
    M = num_pixels + num_padding

    tmp_sino = sino.new_zeros((num_slices, num_angles, M))
    tmp_sino[:, :, num_pad_left:num_pad_left + num_pixels] = sino
    return tmp_sino


def unpad(sino, width):
    pad_left, pad_right = num_pad(width)
    return sino[..., pad_left:pad_left+width]


def cycle_filter(h, width):
    out = h.new_zeros(width)
    out[:len(h)] = h
    out[-len(h) + 1:] = torch.flip(h[1:], dims=(0,))
    return out


def rfft(x):
    return torch.rfft(x, signal_ndim=1, normalized=False)


# def fft_filter(x):
#     # fourier transform of filter
#     fourier_filter = fft(x)
#     # Ensure complex dimension equal to real dimension
#     # (forgot why this is necessary...)
#     fourier_filter = fourier_filter[:, 0][:, None]
#     return fourier_filter


def irfft(x, out_width=None):
    if out_width is None:
        out_width = x.shape[-1]
    return torch.irfft(
        x,
        signal_ndim=1,
        signal_sizes=(out_width,),
        normalized=False
    )


def filter_sino(sino, fourier_filter):
    (num_slices, num_angles, num_pixels) = sino.shape

    fourier_sino = rfft(sino)
    fourier_sino *= fourier_filter

    out_sino = irfft(fourier_sino, out_width=num_pixels)
    return out_sino


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


def cmul(a, b):
    ar, ai = a[..., 0], a[..., 1]
    br, bi = b[..., 0], b[..., 1]

    outr = ar * br - ai * bi
    outi = ar * bi + ai * br

    return torch.stack((outr, outi), dim=-1)


def fbp(A, y, padded=True, filter=None, reject_acyclic_filter=True):
    """Compute the FBP algorithm

    :param A: `tomosipo.operator`
    :param y: `torch.tensor`
    :param padded:
    :param filter:
    :returns:
    :rtype:

    """

    original_width = y.shape[-1]

    if padded:
        y = pad(y)
        # Make filter wider
        if filter is not None:
            filter = cycle_filter(filter, y.shape[-1])

    # Use Ram-Lak filter by default.
    if filter is None:
        filter = ram_lak(y.shape[-1]).to(y.device)

    filter_width = filter.shape[-1]

    # Fourier transform of sinogram and filter.
    y_f = rfft(y)
    h_f = rfft(filter)
    if h_f[..., 1].mean() > ts.epsilon and reject_acyclic_filter:
        raise ValueError(
            "Filter has complex components in Fourier domain. Make sure that the filter is cyclic"
        )
    # Remove complex part of h_f
    h_f = h_f[:, 0:1]
    # Filter the sinogram using "complex multiplication" with real
    # part of h_f:
    y_f *= h_f

    y_filtered = irfft(y_f, filter_width)

    if padded:
        y_filtered = unpad(y_filtered, original_width)
        # By removing the padding, y_filtered has become
        # discontiguous. ASTRA only accepts contiguous arrays. So we
        # allocate a new contiguous array.
        y_filtered = y_filtered.contiguous()

    # Backproject the filtered sinogram to obtain a reconstruction
    rec = A.T(y_filtered)

    # Scale result to make sure that fbp(A, A(x)) == x holds at least
    # to some approximation. In limited experiments, this is true for
    # this version of FBP up to 1%.
    vg, pg = A.volume_geometry, A.projection_geometry
    pixel_area = np.prod(np.array(pg.det_size) / np.array(pg.det_shape))
    voxel_volume = np.prod(np.array(vg.size / np.array(vg.shape)))
    scaling = (np.pi / pg.num_angles) * (pixel_area / voxel_volume) ** 2
    rec *= scaling

    return rec

"""
Self-Flow Utility Functions.

This module contains utility functions for positional encoding and
token processing used in Self-Flow inference.
"""

from typing import Literal, Tuple

import torch
from einops import rearrange
from torch import Tensor


Axes = Tuple[Literal["t", "h", "w", "l"], ...]


def prc_vid(
    x: Tensor, t_coord: Tensor | None = None, l_coord: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    c, t, h, w = x.shape
    if t_coord is None:
        t_coord = torch.arange(t)
    x_coords = {
        "t": t_coord,
        "h": torch.arange(h, device=t_coord.device),
        "w": torch.arange(w, device=t_coord.device),
        "l": torch.arange(1) if l_coord is None else l_coord,
    }
    x_ids = torch.cartesian_prod(
        x_coords["t"], x_coords["h"], x_coords["w"], x_coords["l"]
    )
    x = rearrange(x, "c t h w -> (t h w) c")
    return x, x_ids


def prc_img(
    x: Tensor, t_coord: Tensor | None = None, l_coord: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    c, h, w = x.shape
    x_coords = {
        "t": torch.arange(1) if t_coord is None else t_coord,
        "h": torch.arange(h),
        "w": torch.arange(w),
        "l": torch.arange(1) if l_coord is None else l_coord,
    }
    x_ids = torch.cartesian_prod(
        x_coords["t"], x_coords["h"], x_coords["w"], x_coords["l"]
    )
    x = rearrange(x, "c h w -> (h w) c")
    return x, x_ids


def prc_txt(
    x: Tensor, t_coord: Tensor | None = None, l_coord: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    assert l_coord is None, "l_coord not supported for txts"

    l, c = x.shape

    coords = {
        "t": torch.arange(1) if t_coord is None else t_coord,
        "h": torch.arange(1),
        "w": torch.arange(1),
        "l": torch.arange(l),
    }
    x_ids = torch.cartesian_prod(coords["t"], coords["h"], coords["w"], coords["l"])
    return x, x_ids


def prc_txts(
    x: Tensor, t_coord: Tensor | None = None, l_coord: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    assert l_coord is None, "l_coord not supported for txts"

    t, l, c = x.shape

    if t_coord is None:
        t_coord = torch.arange(t)
    coords = {
        "t": t_coord,
        "h": torch.arange(1, device=t_coord.device),
        "w": torch.arange(1, device=t_coord.device),
        "l": torch.arange(l, device=t_coord.device),
    }
    x_ids = torch.cartesian_prod(coords["t"], coords["h"], coords["w"], coords["l"])
    x = rearrange(x, "t l c -> (t l) c")
    return x, x_ids


def prc_times(t_coord: Tensor) -> Tensor:
    coords = {
        "t": t_coord.to(dtype=int),
        "h": torch.arange(1, device=t_coord.device),
        "w": torch.arange(1, device=t_coord.device),
        "l": torch.arange(1, device=t_coord.device),
    }
    x_ids = torch.cartesian_prod(coords["t"], coords["h"], coords["w"], coords["l"])
    return x_ids


def batched_wrapper(fn):
    def batched_prc(
        x: Tensor, t_coord: Tensor | None = None, l_coord: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        results = []
        for i in range(len(x)):
            results.append(
                fn(
                    x[i],
                    t_coord[i] if t_coord is not None else None,
                    l_coord[i] if l_coord is not None else None,
                )
            )
        x, x_ids = zip(*results)
        return torch.stack(x), torch.stack(x_ids)

    return batched_prc


batched_prc_vid = batched_wrapper(prc_vid)
batched_prc_img = batched_wrapper(prc_img)
batched_prc_txt = batched_wrapper(prc_txt)
batched_prc_txts = batched_wrapper(prc_txts)


def batched_prc_times(t_coord: Tensor) -> Tensor:
    x_ids = []
    for i in range(len(t_coord)):
        x_ids.append(prc_times(t_coord[i]))
    return torch.stack(x_ids)


def compress_time(t_ids: Tensor) -> Tensor:
    """
    Compressing time ids i.e.:
    [0, 0 ... 4, 4 ... 8, 8 ...] ->  [0, 0 ... 1, 1 ... 2, 2 ...]
    """
    assert t_ids.ndim == 1
    t_ids_max = torch.max(t_ids)
    t_remap = torch.zeros((t_ids_max + 1,), device=t_ids.device, dtype=t_ids.dtype)
    t_unique_sorted_ids = torch.unique(t_ids, sorted=True)
    t_remap[t_unique_sorted_ids] = torch.arange(
        len(t_unique_sorted_ids), device=t_ids.device, dtype=t_ids.dtype
    )
    t_ids_compressed = t_remap[t_ids]
    return t_ids_compressed


def scatter_ids(x: Tensor, x_ids: Tensor) -> list[Tensor]:
    """
    Using position ids to scatter tokens into place.
    """
    x_list = []
    t_coords = []
    for data, pos in zip(x, x_ids):
        l, ch = data.shape
        t_ids = pos[:, 0].to(torch.int64)
        h_ids = pos[:, 1].to(torch.int64)
        w_ids = pos[:, 2].to(torch.int64)

        t_ids_cmpr = compress_time(t_ids)

        t = torch.max(t_ids_cmpr) + 1
        h = torch.max(h_ids) + 1
        w = torch.max(w_ids) + 1

        flat_ids = t_ids_cmpr * w * h + h_ids * w + w_ids

        # Optimized: pre-allocate on correct device and use scatter_ for better GPU performance
        out = torch.zeros((t * h * w, ch), device=data.device, dtype=data.dtype)
        out.scatter_(0, flat_ids.unsqueeze(1).expand(-1, ch), data)

        x_list.append(rearrange(out, "(t h w) c -> 1 c t h w", t=t, h=h, w=w))
        t_coords.append(torch.unique(t_ids, sorted=True))
    return x_list


def times_to_ids(time: Tensor) -> Tensor:
    """Using a unit of 10 ms per index."""
    return (time * 1000 // 10).to(dtype=torch.int64)


def ids_to_times(ids: Tensor) -> Tensor:
    """Using a unit of 10 ms per index."""
    return ids * 10 / 1000


def scattercat(x: torch.Tensor, x_ids: torch.Tensor) -> torch.Tensor:
    """Scatter tokens to spatial format and concatenate."""
    x = scatter_ids(x, x_ids)
    return torch.cat(x, 0).squeeze(2)


def scatter_ids_to_times(x_ids: Tensor):
    t_coords = []
    for pos in x_ids:
        t_ids = pos[:, 0].to(torch.int64)
        t_coords.append(ids_to_times(torch.unique(t_ids, sorted=True)))
    return t_coords

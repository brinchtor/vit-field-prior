"""
plot_attention.py  (MATPLOTLIB ONLY -- no torch)

Reads attention_data.npz from compute_attention.py and writes ONE figure per
model, named attention_model_<tag>.pdf (tag = A, B, C, D). Each figure stacks:

    top    : that model's own partial generation at this step (what it sees;
             revealed patches black/white, masked patches grey),
    bottom : that model's log-attention map QK^T/sqrt(dh) + R from the query.

The query patch (current location in the recursion) is outlined in green in
both panels. Both panels are drawn at pixel resolution (H x W) so they align.
Intended to be dropped into a 2x2 LaTeX subfigure block.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

IN_NPZ = "attention_data.npz"
CMAP_ATTN = "viridis"


def model_tag(label):
    return label.split(":")[0].strip().replace(" ", "_")

from mpl_toolkits.axes_grid1 import make_axes_locatable

def make_model_figure(label, attn_map, context_img, query_rc, grid, image_size,
                      patch_size, out_path):
    Gh, Gw = grid
    H, W = image_size
    ps = patch_size
    qr, qc = query_rc

    attn_px = np.kron(attn_map, np.ones((ps, ps)))

    fig, (ax_ctx, ax_att) = plt.subplots(2, 1, figsize=(3.2, 4.2),
                                          gridspec_kw={"hspace": 0.28})

    ax_ctx.imshow(context_img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax_ctx.add_patch(mpatches.Rectangle((qc*ps - 0.5, qr*ps - 0.5), ps, ps,
                                        linewidth=2, edgecolor="lime", facecolor="none"))
    ax_ctx.set_title("Context", fontsize=10)
    ax_ctx.set_xticks([]); ax_ctx.set_yticks([])
    # force a symmetric canvas by adding invisible sentinel patches
    ax_att.set_xlim(-0.5, W - 0.5)   # already true but makes it explicit
    ax_att.set_ylim(H - 0.5, -0.5)
    
    im = ax_att.imshow(attn_px, cmap=CMAP_ATTN, interpolation="nearest",
                       vmin=vmin, vmax=vmax)
    ax_att.add_patch(mpatches.Rectangle((qc*ps - 0.5, qr*ps - 0.5), ps, ps,
                                        linewidth=2, edgecolor="lime", facecolor="none"))
    ax_att.set_title(r"Log-attention  $QK^\top/\sqrt{d_h}+R$", fontsize=10)
    ax_att.set_xticks([]); ax_att.set_yticks([])

    fig.canvas.draw()  # forces layout so get_position() is accurate
    pos = ax_att.get_position()
    cax = fig.add_axes([pos.x1 + 0.02, pos.y0, 0.025, pos.height])
    fig.colorbar(im, cax=cax)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")

if __name__ == "__main__":
    d = np.load(IN_NPZ, allow_pickle=False)
    labels   = [str(s) for s in d["labels"]]
    attn     = d["attn_maps"]
    ctx      = d["context_imgs"]
    query_rc = tuple(int(x) for x in d["query_patch"])
    grid     = tuple(int(x) for x in d["grid"])
    imsize   = tuple(int(x) for x in d["image_size"])
    ps       = int(d["patch_size"])
    
    vmin, vmax = attn.min(), attn.max()

    for i, label in enumerate(labels):
        make_model_figure(label, attn[i], ctx[i], query_rc, grid, imsize, ps,
                          f"attention_model_{model_tag(label)}.pdf")
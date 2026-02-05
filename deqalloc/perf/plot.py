import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
import sys

if len(sys.argv) != 2:
    print("Usage: python3 plot.py <file>")
    sys.exit(1)

# ---------- Load & aggregate ----------
filename = sys.argv[1]
df = pd.read_csv(filename)

# Average repetitions
df = (
    df.groupby(["allocator", "ds", "smr"], as_index=False)
      .agg(value=("value", "mean"))
)

# Order SMRs explicitly (important for stable layout)
smr_order = [
    "2geibr", "2geibr_df",
    "debra", "debra_df",
    "he", "he_df",
    "hp", "hp_df",
    "rcu", "rcu_df",
    "nbr", "nbr_df",
    "nbrplus", "nbrplus_df",
    "qsbr", "qsbr_df",
    "wfe", "wfe_df",
]


df["smr"] = pd.Categorical(df["smr"], smr_order, ordered=True)

# Data structures to plot
ds_list = df["ds"].unique()
allocators = df["allocator"].unique()

# ---------- Plot styling ----------
plt.rcParams.update({
    "font.size": 11,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.5,
    "axes.formatter.use_mathtext": True,
})

mult=1
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12*mult, 2.5*mult))

x = np.arange(len(smr_order)) #* len(allocators)
width = 0.95 / len(ds_list)

ds_hatches = ['..', 'oo', 'OO', 'O.']
allocator_hatches = ['\\\\', '////']

colors = ["orangered", "royalblue", "forestgreen", "gold"]

#repeat color scheme as necessary
colors = np.array([colors] * (int(len(ds_list) / 2+0.5))).flatten()

# ---------- Labels ----------
allocator_handles = [
    Patch(facecolor="white", edgecolor="black", hatch=allocator_hatches[i], label=a)
    for i, a in enumerate(allocators)
]

ds_handles = [
    Patch(facecolor=colors[i], edgecolor="black", hatch=ds_hatches[i % len(ds_hatches)], label=ds)
    for i, ds in enumerate(ds_list)
]

# ---------- Bars ----------
for i, ds in enumerate(ds_list):
    for j, allocator in enumerate(allocators):
        ax_ = axes[j]
        all_df = df[df["allocator"] == allocator]
        y = (
            all_df[all_df["ds"] == ds]
            .set_index("smr")
            .reindex(smr_order)["value"]
        )
        ax_.bar(
            x + i * width,
            y,
            width,
            hatch=ds_hatches[i % len(ds_hatches)] + allocator_hatches[j],
            edgecolor="black",
            alpha=1,
            color=colors[i],
        )

# ---------- Axes ----------

#replace some strings for coherency sake
for i in range(len(smr_order)):
    smr_order[i] = smr_order[i].replace("2geibr", "ibr")
    smr_order[i] = smr_order[i].replace("plus", "+")
    smr_order[i] = smr_order[i].replace("_df", "_af")

ax = axes[0]
ax.set_xticks(x + width * (len(ds_list) - 1) / 2)
ax.set_xticklabels(smr_order, rotation=45, ha="right")
ax.ticklabel_format(axis="y", style="plain", scilimits=(0, 0))

legs = []
legs.append(ax.legend(
    handles=ds_handles,
    frameon=True,
    fontsize=9,
    ncol=len(ds_list),
    loc="upper left",
    alignment="left"
))

for j, allocator in enumerate(allocators):
    ax_ = axes[j]
    #allocator legend
    p = [Patch(
        facecolor="white",
        edgecolor="black",
        hatch=allocator_hatches[j],
        label=allocator)]
    legs.append(ax_.legend(
        handles=p,
        frameon=True,
        fontsize=9,
        ncol=1,
        loc="upper right",
        alignment="left",
    ))
    ax_.set_ylabel("runtime %")
    #margin extend
    ax.set_xlim(x.min() - width, x.max() + width * len(ds_list))
    ax.margins(x=0)
    #grid
    maxy = int((int(ax_.get_ylim()[1]) / 10)) * 10
    ax_.yaxis.set_major_locator(MultipleLocator(maxy / 5))
    ax_.grid(True, which="major", axis="y")

legs.pop()
for l in legs:
    ax.add_artist(l)

plt.tight_layout()
#plt.show()
plt.savefig("perf_analysis.pdf")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import LinearLocator
from matplotlib.ticker import FormatStrFormatter
import re
import sys
import statistics

if len(sys.argv) != 2:
    print("Usage: python3 plot_aggregate.py <file>")
    sys.exit(1)

filename = sys.argv[1]

def parse_line(line):
    line = line.strip()
    if not line or line.startswith("allocator"):
        return None
    parts = line.split()  # split on any whitespace
    allocator = parts[0]
    update_pct = int(parts[1])
    scheme = parts[2]          # now correctly "debra_df" or "nbrplus_df"
    ds = parts[3]
    key_size = int(parts[4])
    # Extract throughput list inside [...]
    throughput_re = re.compile(r"\[\s*([0-9\s]+)\s*\]")
    memory_re = re.compile(r",\s*([0-9]+)\s*KB")
    throughput_match = throughput_re.search(line)
    memory_match = memory_re.search(line)
    if not throughput_match or not memory_match:
        return None
    throughputs = list(map(int, throughput_match.group(1).split()))
    memory = int(memory_match.group(1))
    return {
        "allocator": allocator,
        "update%": update_pct,
        "scheme": scheme.replace("_df", ""),  # normalize here
        "ds": ds,
        "key_size": key_size,
        "throughput": statistics.mean(throughputs),
        "throughput_stdev": statistics.stdev(throughputs),
        "throughputs_raw": throughputs,
        "memory": memory,
    }

rows = []
with open(filename) as f:
    for line in f:
        parsed = parse_line(line)
        if parsed:
            rows.append(parsed)

df = pd.DataFrame(rows)

# Define configuration key
config_cols = ["update%", "scheme", "ds", "key_size"]

df["best_in_config"] = df.groupby(config_cols)["throughput"].transform("max")
# Normalized scoring
df["normalized_score"] = df["throughput"] / df["best_in_config"]
# Winners
df["was_best"] = df["throughput"] == df["best_in_config"]

# Throughput / Memory usage
df["mops_per_kb"] = df["throughput"] / df["memory"]
df["best_in_config_mopspkb"] = df.groupby(config_cols)["mops_per_kb"].transform("max")
df["normalized_mops_per_kb"] = df["mops_per_kb"] / df["best_in_config_mopspkb"]

#normalize throughputs to the best performing allocator in each benchmark and sum normalized values
normalized_sum = df.groupby("allocator", as_index=False)["normalized_score"].sum()
#sum all throughputs for each allocator
absolute_sum = df.groupby("allocator", as_index=False)["throughput"].sum()
#number of times that allocator was the best performing
best_sum = df.groupby("allocator", as_index=False)["was_best"].sum()
#Sum of mop per kb
normalized_mops_per_kb_sum = df.groupby("allocator", as_index=False)["normalized_mops_per_kb"].sum()

#geometric mean
gmean_throughput = df.groupby("allocator")["throughputs_raw"] \
        .apply(lambda s: np.exp(np.mean(np.log(np.concatenate(s.values))))) \
        .rename("gmean_throughput").to_frame()

#gmean_max = gmean_throughput["gmean_throughput"].max()
#normalized_gmean = gmean_throughput / gmean_max

# ---------- Plot ----------
mult=1
fig, ax = plt.subplots(1, 1, figsize=(8*mult, 3*mult))

# ---------- Styling ----------
plt.rcParams.update({
    "font.size": 12,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.5,
    "axes.formatter.use_mathtext": True,
})

allocators = df["allocator"].unique()

hatches = ['\\\\\\\\', '////', 'xxxx', '----', '++', 'xx', '..', 'oo', 'OO']
colors = ["orangered", "royalblue", "forestgreen", "gold"]


# ---------- Labels ----------
handles = [
    Patch(facecolor=colors[0], edgecolor="black", hatch=hatches[0], label="Normalized Throughput Sum"),
    Patch(facecolor=colors[1], edgecolor="black", hatch=hatches[1], label="Normalized Mop/s/KB Sum"),
    Patch(facecolor=colors[2], edgecolor="black", hatch=hatches[2], label="Total Wins"),
    Patch(facecolor=colors[3], edgecolor="black", hatch=hatches[3], label="Geometric Mean of Throughputs"),
]

# ---------- Bars ----------
assert(len(allocators) == len(hatches))

x = np.arange(len(allocators))
width = 0.225

joint_results = normalized_sum \
    .merge(best_sum, on="allocator") \
    .merge(normalized_mops_per_kb_sum, on="allocator") \
    .merge(gmean_throughput, on="allocator")

bars_left = []
bars_left.append(ax.bar(
    x - 1.5 * width,
    joint_results["normalized_score"],
    width,
    label="Normalized sum",
    hatch=hatches[0 % len(hatches)],
    edgecolor="black",
    alpha=1,
    color=colors[0],
))

bars_left.append(ax.bar(
    x - 0.5 * width,
    joint_results["normalized_mops_per_kb"],
    width,
    hatch=hatches[1 % len(hatches)],
    edgecolor="black",
    alpha=1,
    color=colors[1],
))

bars_left.append(ax.bar(
    x + 0.5 * width,
    joint_results["was_best"],
    width,
    hatch=hatches[2 % len(hatches)],
    edgecolor="black",
    alpha=1,
    color=colors[2],
))

bars_right = []
ax_right = ax.twinx()
bars_right.append(ax_right.bar(
    x + 1.5 * width,
    joint_results["gmean_throughput"],
    width,
    hatch=hatches[3 % len(hatches)],
    edgecolor="black",
    alpha=1,
    color=colors[3],
))

def annotate_bars(ax, bar_list, fontsize, fmt):
    for bars in bar_list:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                fmt.format(height),
                ha="center",
                va="bottom",
                fontsize=fontsize,
                fontweight='bold',)

fontsize = 8
annotate_bars(ax, bars_left[0:2],   fontsize * .8, "{:.1f}")
annotate_bars(ax, [bars_left[2]],   fontsize * .8, "{:.0f}")
annotate_bars(ax_right, bars_right, fontsize * .8, "{:.1E}")

ax.legend(
    handles=handles,
    frameon=True,
    fontsize=fontsize,
    ncol=2,
    loc="upper right",
    alignment="left",
    bbox_to_anchor=(1.0, 1.03)
)

#leeway_yaxis = 0.0855
leeway_yaxis = 0.075
ax.set_xticks(x)
ax.set_xticklabels(allocators, rotation=45, ha="right")

for axis in [ax, ax_right]:
        #add room for the labels
        axis.set_ylim(0, axis.get_ylim()[1] * (1 + leeway_yaxis))
        axis.tick_params(labelsize = fontsize)

number_yaxis_ticks = 7

#alight right yaxis to left yaxis with specific formatting -- with Gemini magic
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
for axis in [ax, ax_right]:
    axis.set_ylim(0, axis.get_ylim()[1] * (1 + leeway_yaxis))
ax.yaxis.set_major_locator(MaxNLocator(nbins=number_yaxis_ticks))
ticks_left = ax.get_yticks()
view_max = ax.get_ylim()[1]
ticks_left = ticks_left[(ticks_left >= 0) & (ticks_left <= view_max)]
if len(ticks_left) > 1:
    ratio = ax_right.get_ylim()[1] / ax.get_ylim()[1]
    ax_right.set_yticks(ticks_left * ratio)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax_right.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))
ax.grid(True, linestyle="--", alpha=0.25, axis='y')
ax.tick_params(labelsize=fontsize)
ax_right.grid(False)
ax_right.tick_params(labelsize=fontsize)

#align both yaxis
#ax.yaxis.set_major_locator(LinearLocator(number_yaxis_ticks))
#ax_right.yaxis.set_major_locator(LinearLocator(number_yaxis_ticks))

#x10^7 on right yaxis
#ax_right.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
#ax_right.yaxis.get_offset_text().set_fontsize(fontsize - 2)
#ax_right.yaxis.get_offset_text().set_x(1.0)
#ax_right.yaxis.get_offset_text().set_y(1.005)


plt.tight_layout()
#plt.show()
plt.savefig("aggregate_analysis_setbench.pdf")
plt.savefig("aggregate_analysis_setbench.png")

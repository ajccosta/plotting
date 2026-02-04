import pandas as pd
import matplotlib.pyplot as plt
import re
import sys

if len(sys.argv) != 2:
    print("Usage: python3 plot_aggregate.py allocator_benchmarks.data")
    sys.exit(1)

filename = sys.argv[1]

rows = []
with open(filename) as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("allocator"):
            continue

        parts = re.split(r"\s{2,}|\t", line)
        if len(parts) < 7:
            continue

        allocator = parts[0]
        update_pct = int(parts[1])
        scheme = parts[2]
        ds = parts[3]
        key_size = int(parts[4])

        # Extract mean throughput before the comma
        mean_match = re.search(r"\]\s*([\d\.]+),", parts[6])
        if not mean_match:
            continue
        mean_throughput = float(mean_match.group(1))

        rows.append({
            "allocator": allocator,
            "update%": update_pct,
            "scheme": scheme,
            "ds": ds,
            "key_size": key_size,
            "throughput": mean_throughput,
        })

df = pd.DataFrame(rows)

# Define configuration key
config_cols = ["update%", "scheme", "ds", "key_size"]

#remove _df suffix from schemes
df["scheme"] = df["scheme"].apply(lambda x: x.replace("_df", ""))

# Normalized scoring
df["best_in_config"] = df.groupby(config_cols)["throughput"].transform("max")
df["normalized_score"] = df["throughput"] / df["best_in_config"]
df["was_best"] = df["throughput"] == df["best_in_config"]

normalized_sum = df.groupby("allocator")["normalized_score"].sum().sort_values(ascending=False)
absolute_sum = df.groupby("allocator")["throughput"].sum().sort_values(ascending=False)
#number of times that allocator was the best performing
best_sum = df.groupby("allocator")["was_best"].sum().sort_values(ascending=False)

# Plot normalized scores
plt.figure()
normalized_sum.plot(kind="bar")
plt.ylabel("Sum of Normalized Scores (0–1 per configuration)")
plt.title("Allocator Performance (Normalized per Configuration)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Plot absolute throughput
plt.figure()
absolute_sum.plot(kind="bar")
plt.ylabel("Sum of Absolute Throughput")
plt.title("Allocator Performance (Absolute Throughput Sum)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Plot winners
plt.figure()
best_sum.plot(kind="bar")
plt.ylabel("Number of times the allocator was the best performing")
plt.title("Total Wins")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Print tables
print("\n=== Normalized Scores ===")
print(normalized_sum)

print("\n=== Absolute Throughput Sum ===")
print(absolute_sum)

print("\n=== Number of Wins ===")
print(best_sum)

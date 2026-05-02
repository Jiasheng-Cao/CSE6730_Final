import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# ------------------------------------------------------------
# 1. Input data
# ------------------------------------------------------------

data = [
    # NH4_inf, NH4_exp, NO2_exp, NO3_exp, NH4_sim, NO2_sim, NO3_sim
    [13.4, 0.8, 0.2, 12.7, 1.4227579316682766, 1.6727186554437503, 10.304523412887988],
    [13.3, 0.0, 0.7, 12.9, 1.4227540545590422, 1.6727129743210870, 10.204532971119908],
    [11.3, 0.0, 0.9, 10.9, 1.4226765584621242, 1.6726052530268933, 8.204718189000000],
    [18.1, 0.0, 2.2, 15.6, 1.4229403403778231, 1.6729879544778030, 15.004071705144401],
    [18.8, 0.2, 3.4, 14.9, 1.4229675383655462, 1.6730281462665290, 15.704004315367923],
    [13.8, 0.0, 0.0, 14.7, 1.4227734417603235, 1.6727414462282013, 10.704485110000000],
    [13.5, 1.8, 0.3, 13.0, 1.4227618089438720, 1.6727243441935764, 10.404513846862560],
    [16.1, 2.6, 2.1, 12.6, 1.4228626755992204, 1.6728731888837670, 13.004264135517015],
    [16.5, 0.6, 0.3, 15.2, 1.4228782033521350, 1.6728961338190618, 13.404225662828804],
    [20.2, 6.9, 1.9, 12.3, 1.4230219580685053, 1.6731085670031158, 17.103869474928440],
    [15.9, 1.4, 0.0, 17.7, 1.4228549127016350, 1.6728617181163747, 12.804283369181999],
    [14.6, 0.2, 0.0, 15.2, 1.4228044698065858, 1.6727872017305696, 11.504408328462840],
    [17.1, 0.3, 0.0, 16.6, 1.4229014998667122, 1.6729305590350040, 14.004167940000000],
    [17.3, 0.3, 0.1, 16.8, 1.4229092666717862, 1.6729420361199576, 14.204148697208280],
    [14.8, 0.2, 1.2, 14.0, 1.4228122284526703, 1.6727986582409553, 11.704389113306360],
    [15.1, 0.3, 0.0, 16.5, 1.4228238676485760, 1.6728158497810428, 12.004360282570340],
    [14.5, 0.0, 0.0, 18.4, 1.4228005907288073, 1.6727814752546060, 11.404417934016594],
    [11.0, 0.0, 3.1, 7.7, 1.4226649493954882, 1.6725918150340370, 7.9047432355704705],
    [11.3, 0.0, 0.0, 12.9, 1.4226765584621242, 1.6726052530268933, 8.204718189000000],
    [10.5, 0.0, 3.4, 6.6, 1.4226456287632374, 1.6725735527678849, 7.404780818468861],
    [13.4, 0.8, 2.7, 10.8, 1.4227579316682766, 1.6727186554437503, 10.304523412887988],
    [11.5, 0.0, 2.3, 7.1, 1.4226843018312954, 1.6726149164003410, 8.404700781768339],
]

df = pd.DataFrame(
    data,
    columns=[
        "NH4_inf",
        "NH4_exp", "NO2_exp", "NO3_exp",
        "NH4_sim", "NO2_sim", "NO3_sim"
    ]
)

# ------------------------------------------------------------
# 2. Calculate error metrics
# ------------------------------------------------------------

def calculate_metrics(exp, sim):
    """
    Error is defined as:
        error = simulation - experiment
    """
    exp = np.array(exp, dtype=float)
    sim = np.array(sim, dtype=float)
    error = sim - exp

    n = len(error)
    mean_error = np.mean(error)
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error ** 2))

    # 95% confidence interval of mean error using t-distribution
    sd = np.std(error, ddof=1)
    se = sd / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    ci_low = mean_error - t_crit * se
    ci_high = mean_error + t_crit * se

    # R2 = 1 - SSE/SST
    sse = np.sum((sim - exp) ** 2)
    sst = np.sum((exp - np.mean(exp)) ** 2)
    r2 = 1 - sse / sst

    return mean_error, ci_low, ci_high, mae, rmse, r2


results = []

for species in ["NH4", "NO2", "NO3"]:
    mean_error, ci_low, ci_high, mae, rmse, r2 = calculate_metrics(
        df[f"{species}_exp"],
        df[f"{species}_sim"]
    )

    results.append({
        "Species": species,
        "Mean Error": mean_error,
        "95% CI Low": ci_low,
        "95% CI High": ci_high,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

metrics_df = pd.DataFrame(results)

print("\nError metrics:")
print(metrics_df.round(3))

# Save metrics table as CSV
metrics_df.to_csv("error_metrics.csv", index=False)

# ------------------------------------------------------------
# 3. Generate LaTeX table
# ------------------------------------------------------------

def species_latex_name(species):
    if species == "NH4":
        return r"NH$_4^+$"
    if species == "NO2":
        return r"NO$_2^-$"
    if species == "NO3":
        return r"NO$_3^-$"
    return species


latex_lines = []
latex_lines.append(r"\begin{table}[H]")
latex_lines.append(r"\centering")
latex_lines.append(r"\caption{Error metrics for effluent nitrogen species across the validation dataset}")
latex_lines.append(r"\label{tab:error_metrics}")
latex_lines.append(r"\begin{tabular}{c|c|c|c|c|c}")
latex_lines.append(r"\hline")
latex_lines.append(r"Species & Mean Error & 95\% CI & MAE & RMSE & $R^2$ \\")
latex_lines.append(r" & (mg N/L) & (mg N/L) & (mg N/L) & (mg N/L) &  \\")
latex_lines.append(r"\hline")

for _, row in metrics_df.iterrows():
    species = species_latex_name(row["Species"])
    ci_text = f"[{row['95% CI Low']:.3f}, {row['95% CI High']:.3f}]"
    latex_lines.append(
        f"{species} & {row['Mean Error']:.3f} & {ci_text} & "
        f"{row['MAE']:.3f} & {row['RMSE']:.3f} & {row['R2']:.3f} \\\\"
    )

latex_lines.append(r"\hline")
latex_lines.append(r"\end{tabular}")
latex_lines.append(r"\end{table}")

latex_table = "\n".join(latex_lines)

print("\nLaTeX table:")
print(latex_table)

with open("error_metrics_table.tex", "w") as f:
    f.write(latex_table)

# ------------------------------------------------------------
# 4. Generate parity plot
# ------------------------------------------------------------

plt.figure(figsize=(7, 6))

plt.scatter(df["NH4_exp"], df["NH4_sim"], marker="o", label="NH$_4^+$-N")
plt.scatter(df["NO2_exp"], df["NO2_sim"], marker="s", label="NO$_2^-$-N")
plt.scatter(df["NO3_exp"], df["NO3_sim"], marker="^", label="NO$_3^-$-N")

all_exp = np.concatenate([
    df["NH4_exp"].values,
    df["NO2_exp"].values,
    df["NO3_exp"].values
])

all_sim = np.concatenate([
    df["NH4_sim"].values,
    df["NO2_sim"].values,
    df["NO3_sim"].values
])

min_val = min(all_exp.min(), all_sim.min()) - 0.5
max_val = max(all_exp.max(), all_sim.max()) + 0.5

plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", label="1:1 line")

plt.xlim(min_val, max_val)
plt.ylim(min_val, max_val)
plt.xlabel("Experimental concentration (mg N/L)")
plt.ylabel("Simulated concentration (mg N/L)")
plt.title("Parity Plot for Effluent Nitrogen Species")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("effluent_nitrogen_parity_plot.png", dpi=300)
plt.show()

print("\nSaved files:")
print(" - error_metrics.csv")
print(" - error_metrics_table.tex")
print(" - effluent_nitrogen_parity_plot.png")

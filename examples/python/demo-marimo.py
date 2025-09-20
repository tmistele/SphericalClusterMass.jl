import marimo

__generated_with = "0.16.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    import numpy as np
    import astropy.units as u
    import astropy.constants as const
    from astropy.visualization import quantity_support
    quantity_support()
    import matplotlib.pyplot as plt

    # That's our own code
    import deproject as d
    return d, mo, np, plt, u


@app.cell(hide_code=True)
def _():
    # Do the deprojection
    return


@app.cell
def _(d, np, u):
    R = np.array([0.1, 0.2, 0.3, .4]) * u.Mpc
    G = 1e3 * np.array([0.3, 0.2, 0.1, .05]) * u.Msun / u.pc**2
    f = 1e-3 * np.array([0.9, 0.9, 0.9, 0.9]) / (u.Msun / u.pc**2)
    G_covariance = np.diag((G * 0.1) ** 2)  # 10% measurement error

    result = d.calculate_M_and_covariance_in_bins(
        R=R,
        G=G,
        G_covariance=G_covariance,
        f=f,
        extrapolate=d.jl.ExtrapolatePowerDecay(1),
        interpolate=d.jl.InterpolateR(1),
    )

    # Correlation matrix
    M_stat_corr = result.M_stat_cov / np.outer(result.M_stat_err, result.M_stat_err)
    return M_stat_corr, R, result


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Plot results""")
    return


@app.cell
def _(R, plt, result):
    plt.errorbar(R, result.M, yerr=result.M_stat_err)
    plt.show()
    return


@app.cell
def _(M_stat_corr, R, plt):
    plt.matshow(M_stat_corr)
    plt.gca().set_xticks(range(R.shape[0]), labels=R)
    plt.gca().set_yticks(range(R.shape[0]), labels=R)
    plt.show()
    return


@app.cell
def _(M_stat_corr):
    M_stat_corr.value
    return


if __name__ == "__main__":
    app.run()

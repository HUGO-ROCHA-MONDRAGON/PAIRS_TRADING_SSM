import matplotlib.pyplot as plt

def plot_spread_with_signals(spread, U, L, C=None, signal=None, title=""):
    plt.figure(figsize=(10,4))
    plt.plot(spread.index, spread, label="spread")
    plt.axhline(U, linestyle="--", label="U")
    plt.axhline(L, linestyle="--", label="L")
    if C is not None:
        plt.axhline(C, linestyle="-", label="mean")

    if signal is not None:
        changes = signal.diff().fillna(0) != 0
        plt.scatter(spread.index[changes], spread[changes], s=15, label="signal change")

    plt.legend()
    plt.title(title)
    plt.show()



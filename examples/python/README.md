# Examples of usage in Python

These examples use [juliacall](https://juliapy.github.io/PythonCall.jl/stable/), which will install and run julia behind your back.
Since julia is just-in-time-compiled, calling a julia function for the first time can take a while.
After that, things should run fast.

I'm using [`uv`](https://docs.astral.sh/uv/) here but that's not required.

For an example _jupyter_ notebook, see `demo-jupyter.ipynb`.
To try it, you can run from within this directory
```
uv run --with jupyter jupyter lab demo-jupyter.ipynb
```

For an example [_marimo_](https://marimo.io/) notebook, see `demo-marimo.py`.
To try it, you can run from within this directory
```
uv run --with marimo marimo edit demo-marimo.py
```

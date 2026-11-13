# Rare Earth Separation Lab

This is a four-page [marimo](https://marimo.io) notebook (`notebook.py`) about separating La, Ce, Pr, Nd, Sm and Y by
solvent extraction with PC88A / D2EHPA. It is split evenly between two halves:

* **Part A, Separations:** liquid–liquid extraction basics, equilibrium, Fenske/Kremser staging, McCabe–Thiele
  with loading, and the extraction/scrub cascade.
* **Part B, Process dynamics and control:** mixer-settler hydraulics and stage efficiency, start-up, step-test
  identification (FOPDT and SIMC), and PI disturbance rejection.

The four pages are:

1. **Concepts and governing equations:** background on liquid–liquid extraction, plus a concept section for every lab section
2. **Interactive lab:** sections 1–5 (Part A) and 6–9 (Part B)
3. **Python code:** every function in full, plus an in-notebook test suite with a pass/fail table
4. **Attribution:** an AI-generated project; lists models, services and references, with an author placeholder

Dependencies are declared inline (PEP 723): `marimo`, `numpy`, `matplotlib`.

## Open in molab

Replace `<username>` and `<repo>` with your GitHub user name and repository name:

```
https://molab.marimo.io/github/<username>/<repo>/blob/main/notebook.py
```

## Run locally

```bash
uvx marimo run --sandbox notebook.py
```

To edit the notebook instead of only running it:

```bash
uvx marimo edit --sandbox notebook.py
```

## Export to WebAssembly (static site)

```bash
marimo export html-wasm notebook.py -o docs --mode run
```

To preview the export locally:

```bash
python -m http.server --directory docs
```

## Publish with GitHub Pages

1. Commit and push `notebook.py` and the `docs/` folder (including `docs/.nojekyll`) to `main`.
2. On GitHub, open **Settings → Pages**.
3. Under **Build and deployment**, set **Source** to **Deploy from a branch**.
4. Choose **Branch: `main`** and **Folder: `/docs`**, then click **Save**.
5. After a minute or two the site is live at `https://<username>.github.io/<repo>/`.
6. Re-run the export command and push again whenever `notebook.py` changes.

The page loads Pyodide and the Python packages from a CDN on first visit, so the first load takes some time.

## Status

This is a teaching model. The Sm/Nd and Y/Sm separation factors, the Ce(IV) shift, the mass-transfer time,
the volumes and the controller limits are illustrative estimates. See page 4 of the notebook for sources and caveats.

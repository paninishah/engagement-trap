# engagement-trap

**The Engagement Trap** — a movie recommender on MovieLens (ml-latest-small), and a simulation of what happens when a recommender keeps retraining on the clicks it caused. ML Lab mini project (DJS23CPC301L).

**In one sentence:** if you only measure clicks, you can't tell that your recommender is showing people a tiny slice of what's available. You have to measure that separately, and a simple diversity re-ranking (MMR) doubles it at no measurable cost in clicks.

**Main findings** (details and all numbers in [`kaggle_outputs/RESULTS.md`](kaggle_outputs/RESULTS.md)):
1. **The best rating predictor is not the best recommender.** ItemKNN has the lowest RMSE but near-useless top-10 lists; SVD gives the best lists on test.
2. **Recommenders concentrate on popular movies.** SVD ever recommends about 15% of the catalogue. MMR re-ranking widens this with a small, tunable accuracy cost. K-Means clusters work for capping; DBSCAN finds no usable clusters in the movie embeddings.
3. **In a 30-round feedback-loop simulation, clicks are blind to concentration.**
   - Plain SVD shows only about 6% of the catalogue ever, and that barely grows: exposure stays stuck rather than collapsing.
   - MMR shows about 11%.
   - Click rates and simulated satisfaction are the same for both.

## Notebooks

| Notebook | What it does | Lab experiments used |
|---|---|---|
| `notebooks/01_eda_preprocessing.ipynb` | Data checks, EDA, long-tail analysis, features, split check | PCA (Exp 5) preview |
| `notebooks/02_baselines.ipynb` | 4 recommenders + popularity reference, tuned on validation, 3 seeds | Linear Regression (1), SVD (6), PCA (5), KNN (8) |
| `notebooks/03_diversity_rerank.ipynb` | MMR and cluster-cap re-ranking of SVD; K-Means vs DBSCAN | K-Means vs DBSCAN (10) |
| `notebooks/04_feedback_loop.ipynb` | Feedback-loop simulation with a hidden logistic-regression user model | Logistic Regression (2) |
| `notebooks/etlib.py` | All shared code (data, models, metrics, re-rankers, oracle, simulator) | — |

Everything runs on CPU. No GPU is needed.

## Running on Kaggle

**One-time setup**
1. Upload the library: Kaggle → *Datasets* → *New Dataset* → upload `notebooks/etlib.py` → name it e.g. `engagement-trap-lib`. Keep it private if you want.
2. Get the dataset on Kaggle: attach the MovieLens **ml-latest-small** dataset (folder `ml-latest-small/` with `ratings.csv` and `movies.csv`). If you can't find it, upload `ml-latest-small.zip` from GroupLens as your own Kaggle dataset. The notebooks search `/kaggle/input` for `ratings.csv`, so the dataset name and folder layout don't matter.

**For each notebook (01 → 04)**
1. Kaggle → *Code* → *New Notebook* → *File* → *Import Notebook* → upload the `.ipynb`.
2. *Add Input* → attach **your `engagement-trap-lib` dataset** and **the ml-latest-small dataset**.
3. Settings: Accelerator **None** (CPU). Internet can stay off if the MovieLens dataset is attached.
4. *Save Version* → **Save & Run All (Commit)**. This runs the notebook top to bottom and saves the outputs with it.
5. Optional for notebook 04: also attach **notebook 03's output** (*Add Input* → *Your Work* → notebook 03). Notebook 04 then uses the λ and cluster caps that notebook 03 chose on validation. Without it, notebook 04 uses λ = 0.7 and cap = 2 and prints that it did.

**Outputs** (in each notebook's *Output* tab, under `/kaggle/working/`)
- `figures/*.png`: every plot.
- `results/*.csv`: per-seed and mean ± std tables.
- `demo_data/*.json`: data for the demo site (`movies.json`, `baselines.json`, `item_factors.json`, `clusters.json`, `rerank.json`, `simulation.json`).

If you edit `etlib.py`, upload it as a new version of the `engagement-trap-lib` dataset before re-running.

## Demo site (`docs/`)

A static page with no build step and no server: `index.html`, `style.css`, `app.js`, plus the JSON files the notebooks exported (in `docs/data/`). Visitors pick films they like. The browser folds them into the SVD model and shows the plain top-10 next to a diversity-aware (MMR or K-Means-cap) top-10. Below that come the simulation and model-comparison charts.

- **GitHub Pages:** repo *Settings* → *Pages* → *Deploy from a branch* → `main`, folder `/docs`.
- **Vercel:** import the repo and set the root directory to `docs` (no framework, no build command).
- **Locally:** `python -m http.server 8000 --directory docs`, then open http://localhost:8000. Opening `index.html` directly won't load the data.

To refresh the page after re-running the notebooks, copy each notebook's `demo_data/*.json` into `docs/data/`.

## Running locally (optional)

```bash
pip install -r requirements.txt
cd notebooks
jupyter notebook
```

Locally, put the unzipped `ml-latest-small/` folder in `data/` (repo root), or the notebooks download it from GroupLens into `notebooks/data/`. Outputs go to `notebooks/outputs/`. Both folders are git-ignored.

## Data licence

ml-latest-small is © GroupLens Research. Its licence allows redistribution under the same terms, non-commercial use only, with citation and no implied endorsement. Raw data files are still git-ignored to keep the repo small. The demo publishes movie titles/genres and learned model parameters, with attribution. GroupLens calls ml-latest-small a *development* dataset that may change over time; note this in the report's limitations.

Preprocessing: movies with fewer than 5 ratings are dropped before modelling (3,650 of 9,724 rated movies remain, with 90% of ratings). EDA in notebook 01 also covers the unfiltered data.

> F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM TiiS 5(4), Article 19.

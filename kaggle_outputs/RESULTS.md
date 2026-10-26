# Results — The Engagement Trap

Final Kaggle run of all four notebooks. Every number below is copied from the CSV/JSON files in this folder. Each notebook's outputs are in its own sub-folder (`figures/`, `results/`, `demo_data/`).

## The project in one paragraph

We built movie recommenders on MovieLens and compared four algorithms. The model that best predicts star ratings (ItemKNN) makes the worst top-10 lists, while SVD makes the best ones. Even SVD keeps recommending a small set of popular movies, and a diversity re-ranking step (MMR) widens that at a small, adjustable accuracy cost. Finally, we simulated 30 rounds of users clicking recommendations and the system retraining on those clicks. Plain SVD showed only 5.6% of the catalogue in total, and that barely grew: exposure stays stuck rather than collapsing. With MMR it was 10.8%, while click rate and simulated satisfaction stayed the same. **The click metric cannot see the difference**, which is the project's Goodhart point: the number being optimised gives no warning about how narrow the system is, so that has to be measured separately.

What the data does **not** show, and should not be claimed:
- Exposure did not shrink over time.
- Clicks and true satisfaction did not drift apart.
- More diversity did not help users find more of their best-matching movies.

**Setup.** MovieLens ml-latest-small, movies with ≥ 5 ratings (610 users, 3,650 movies, 90,274 ratings). 3 seeds; per-user 80/20 train/test split; hyperparameters tuned on a validation split carved from train (selection metric NDCG@10). Results are mean ± std over the 3 seeds, on **test**. Top-10 lists; a test movie is relevant if rated ≥ 4.0.

---

## 1. Data (notebook 01)

| | Value |
|---|---|
| Raw ratings / rated movies | 100,836 / 9,724 |
| Raw movies with exactly 1 rating | 3,446 (35%) |
| Share of all ratings held by the top 10% of movies (raw) | **60.0%** |
| Gini of ratings per movie: raw → after ≥ 5 filter | 0.716 → 0.529 |
| After filtering: users / movies / ratings | 610 / 3,650 / 90,274 |
| Sparsity (share of user–movie pairs not rated) | 95.9% |
| Median ratings per user / per movie | 68 / 13 |
| Mean rating; share rated ≥ 4.0 | 3.54; 49.6% |
| Correlation of log(popularity) with mean rating | 0.27 |
| Rating period | 1996-03-29 → 2018-09-24 |

**Takeaways.**
- Movie popularity is a steep long tail: the top tenth of movies get six in ten ratings.
- Popular movies are also rated somewhat higher (r = 0.27). So favouring them is partly "justified" by accuracy, which is why accuracy alone can't tell us whether popularity bias is a problem.

Figures: `01_long_tail.png`, `01_ratings.png`, `01_genres.png`, `01_years.png`, `01_pca_content.png`.

---

## 2. Baselines (notebook 02)

| Model | RMSE ↓ | MAE ↓ | Precision@10 | Recall@10 | NDCG@10 | Coverage | Gini ↓ | ILD | AvgPop ↓ | Chosen setting |
|---|---|---|---|---|---|---|---|---|---|---|
| Popularity *(reference)* | 0.9532 ± 0.0121 | 0.7430 ± 0.0069 | 0.1199 ± 0.0047 | 0.1053 ± 0.0036 | 0.1602 ± 0.0038 | 0.0146 ± 0.0004 | 0.9952 | 0.7016 ± 0.0178 | 195.6 ± 1.4 | — |
| LinearRegression (Ridge bias) | 0.9713 ± 0.0107 | 0.7702 ± 0.0070 | 0.1184 ± 0.0042 | 0.0990 ± 0.0031 | 0.1567 ± 0.0065 | 0.0132 ± 0.0007 | 0.9953 | 0.6810 ± 0.0027 | 186.0 ± 0.6 | α = 500 (all seeds) |
| ItemKNN | **0.8802 ± 0.0102** | **0.6602 ± 0.0074** | 0.0205 ± 0.0002 | 0.0091 ± 0.0005 | 0.0219 ± 0.0003 | **0.6511 ± 0.0051** | **0.6173** | 0.7141 ± 0.0056 | **19.7 ± 0.3** | k = 5 (all seeds) |
| SVD | 0.9170 ± 0.0077 | 0.7088 ± 0.0051 | **0.1207 ± 0.0038** | **0.1109 ± 0.0014** | **0.1671 ± 0.0014** | 0.1503 ± 0.0042 | 0.9554 | 0.7099 ± 0.0038 | 123.6 ± 0.9 | k = 40 (all seeds) |
| ContentPCA | 0.9044 ± 0.0080 | 0.6998 ± 0.0048 | 0.0067 ± 0.0011 | 0.0048 ± 0.0009 | 0.0077 ± 0.0013 | 0.3435 ± 0.0430 | 0.8362 | 0.2548 ± 0.0188 | 21.1 ± 0.4 | (8 comps, α 10) ×2, (12, 10) ×1 |

### Which model is "best"? Report this honestly
| | Mean validation NDCG@10 (used for selection) | Test NDCG@10 |
|---|---|---|
| LinearRegression | **0.1206** | 0.1567 |
| SVD | 0.1140 | **0.1671** |

- **By the selection rule fixed in advance (validation NDCG@10), the best model is LinearRegression.** The notebook printed this, and `baselines.json` records it.
- **On test, SVD is best.** LinearRegression scores *below* the Popularity reference (0.1567 vs 0.1602).
- Do **not** switch the rule now that test results are known; that would be tuning on test. Report both lines and explain the gap. At α = 500 the Ridge model is almost a popularity ranker: its biases for rarely-rated movies are shrunk to zero, so it gives every user nearly the same list (Coverage 1.3%, AvgPop 186 vs Popularity's 196).
- Its validation NDCG was still creeping up at the edge of the grid (α = 250 → 500: 0.1194 → 0.1206, +1%). It is approaching popularity ranking, not improving towards a better optimum.
- The validation split comes from a smaller training set (64% of each user's ratings vs 80%), where popularity-style ranking does relatively better than personalised SVD.
- **Why notebooks 03–04 build on SVD anyway:** this is a design reason, not "SVD won". The re-rankers need *personalised* scores and *movie embeddings*. LinearRegression gives everyone the same list and has no embeddings, so there is nothing to re-rank or cluster.

### Findings
1. **The best rating predictor is the worst ranker.**
   - ItemKNN has the lowest RMSE (0.880) but NDCG@10 of only 0.022.
   - ContentPCA has the 2nd-best RMSE and the lowest NDCG (0.008).
   - SVD ranks well despite middling RMSE.
   - The ranking flips depending on the metric. This matches Cremonesi, Koren & Turrin (RecSys 2010), and it is the project's theme in miniature: the number you optimise is not the outcome you want.
2. **Why KNN and ContentPCA rank badly:** they rank by predicted rating, which is extreme and noisy for rarely-rated movies. Those obscure movies fill the top 10 (AvgPop ≈ 20) but rarely appear in users' test sets. Their high Coverage comes from noise, not useful diversity.
3. **Popularity is a strong baseline.** It reaches NDCG 0.160, only 4% below SVD, while showing 1.5% of the catalogue (≈ 53 of 3,650 movies) with exposure Gini 0.995.
4. **SVD's top-10 leans heavily on popular movies.** Recommended movies average 124 training ratings each, (training split), against a median of 13 ratings per movie in the full filtered data (notebook 01), and only 15% of the catalogue (≈ 550 movies) is ever recommended.
5. **ContentPCA has very low genre diversity (ILD 0.25 vs 0.68–0.71 for the others):** it recommends more of the genres you already rated.
6. **Tuning trade-off:** because models were tuned for NDCG, not RMSE, the chosen LinearRegression and KNN settings have worse RMSE than other settings on their grids. For example, KNN k = 40 has validation RMSE 0.849 vs 0.888 at the chosen k = 5 (`02_tuning_validation.csv`).

Figures: `02_baselines_bars.png`, `02_accuracy_vs_popularity.png`.

---

## 3. Diversity-aware re-ranking of SVD (notebook 03)

**Choice rule:** the most diverse setting (highest validation ILD) that keeps ≥ 95% of plain-SVD validation NDCG@10.

| Seed | SVD k | MMR λ | K-Means cap | DBSCAN cap |
|---|---|---|---|---|
| 0 | 40 | 0.7 | 3 † | 1 |
| 1 | 40 | 0.7 | 3 † | 1 |
| 2 | 40 | 0.8 | 3 † | 2 |

† No K-Means cap met the 95% rule (best: cap 3 kept 0.1058 / 0.1140 = 93%), so the fallback, the loosest cap, was used. State this.

| Policy | Precision@10 | NDCG@10 | Coverage ↑ | Gini ↓ | ILD ↑ | AvgPop ↓ |
|---|---|---|---|---|---|---|
| SVD (plain) | 0.1207 ± 0.0038 | **0.1671 ± 0.0014** | 0.1503 ± 0.0042 | 0.9554 ± 0.0008 | 0.7099 ± 0.0038 | 123.6 ± 0.9 |
| **MMR** (λ 0.7/0.7/0.8) | 0.1139 ± 0.0066 | 0.1596 ± 0.0024 | 0.1611 ± 0.0046 | 0.9493 ± 0.0022 | **0.7208 ± 0.0077** | 118.3 ± 1.5 |
| K-Means cap (cap 3) | 0.1063 ± 0.0023 | 0.1545 ± 0.0056 | **0.1655 ± 0.0086** | **0.9455 ± 0.0043** | 0.7177 ± 0.0064 | **110.3 ± 6.5** |
| DBSCAN cap (cap 1/1/2) | 0.1038 ± 0.0189 | 0.1498 ± 0.0149 | 0.1132 ± 0.0327 | 0.9633 ± 0.0067 | 0.7162 ± 0.0094 | 115.6 ± 9.3 |
| Popularity *(reference)* | 0.1199 ± 0.0047 | 0.1602 ± 0.0038 | 0.0146 ± 0.0004 | 0.9952 | 0.7016 ± 0.0178 | 195.6 ± 1.4 |

Relative to plain SVD:
- **MMR:** NDCG −4.5%, Coverage +7%, AvgPop −4%.
- **K-Means cap:** NDCG −7.5%, Coverage +10%, AvgPop −11%.
- **DBSCAN cap:** NDCG −10%, and Coverage **−25%** (it makes concentration worse).

**MMR trade-off curve** (test, mean of 3 seeds):

| λ | 1.0 | 0.9 | 0.8 | 0.7 | 0.6 | 0.5 | 0.4 | 0.3 |
|---|---|---|---|---|---|---|---|---|
| NDCG@10 | 0.1671 | 0.1651 | 0.1621 | 0.1578 | 0.1501 | 0.1380 | 0.1243 | 0.1084 |
| Coverage | 0.150 | 0.151 | 0.155 | 0.164 | 0.178 | 0.198 | 0.225 | 0.249 |
| AvgPop | 123.6 | 122.5 | 120.6 | 117.4 | 111.9 | 104.0 | 94.0 | 84.1 |
| ILD | 0.710 | 0.713 | 0.717 | 0.723 | 0.730 | 0.736 | 0.740 | 0.743 |

**K-Means vs DBSCAN (Exp 10), seed-0 SVD movie embeddings:**

| K-Means k | 5 | 10 | 20 | 30 | **40** | 50 |
|---|---|---|---|---|---|---|
| silhouette (cosine) | 0.126 | 0.183 | 0.243 | 0.292 | **0.303** | 0.283 |

| DBSCAN eps | 0.05 | 0.10 | 0.15 | 0.20 | **0.25** | ≥ 0.30 |
|---|---|---|---|---|---|---|
| clusters | 33 | 41 | 21 | 16 | **2** | 1 |
| noise share | 86% | 67% | 49% | 30% | **14%** | ≤ 4% |
| silhouette | 0.754 | 0.231 | −0.128 | −0.256 | 0.002 | — |

### Findings
1. **MMR gives a smooth, tunable trade-off.** λ = 0.9–0.8 costs very little accuracy, and accuracy falls fast below λ ≈ 0.6. At λ = 0.7 MMR ranks about as well as Popularity (0.158 vs 0.160) while covering 11× more of the catalogue.
2. **Genre diversity barely moves** (ILD 0.710 → 0.743 even at λ = 0.3). Re-ranking mainly changes *how popular* the movies are, not their genres. ILD uses genres while MMR uses SVD embeddings, so this measurement isn't self-fulfilling.
3. **K-Means cap pushes harder against popularity than MMR at a similar NDCG** (cap 3: NDCG 0.155, AvgPop 110; MMR λ = 0.7: 0.158, 117). For the same coverage it costs a bit more accuracy. It never met the 95% rule.
4. **DBSCAN does not suit these embeddings.**
   - Every eps either labels most movies noise (eps ≤ 0.15) or merges almost everything into one cluster (eps ≥ 0.25).
   - The one setting with a high silhouette (eps 0.05, 0.754) labels 86% of movies as noise.
   - SVD embeddings form one continuous cloud with no dense, separated groups, which is the structure DBSCAN assumes.
   - Capping one giant cluster mostly admits the same few hundred noise movies for everyone, so coverage falls and results vary a lot across seeds (NDCG std 0.015, about 10× SVD's).
   - K-Means always splits the space, so it is usable here. Its k = 40 sits inside the grid (silhouette peaks then falls).

Figures: `03_tradeoff.png`, `03_clusters_2d.png`, `03_cluster_selection.png`.

---

## 4. Feedback-loop simulation (notebook 04)

Config: 30 rounds, top-10, implicit-feedback SVD (k = 20) retrained every round, candidate pool 50, start history 10 movies per user. λ = 0.7, K-Means cap 3 and DBSCAN cap 1, taken from notebook 03's output.

**Hidden oracle (Logistic Regression).** It does not use SVD factors.

| Seed | AUC on train | AUC on held-out test |
|---|---|---|
| 0 | 0.824 | 0.786 |
| 1 | 0.824 | 0.782 |
| 2 | 0.823 | 0.788 |

- It is well calibrated on held-out data (`04_oracle.png`).
- The largest coefficients are user–genre affinity terms and the movie's mean rating. The direct log-popularity weight is close to zero, so the oracle does not *prefer* popular movies for being popular.

**Round 30 (mean ± std over 3 seeds):**

| Policy | CTR (proxy) | TrueUtility | Discovery | CumCoverage | ExposureGini | ILD | AvgPop |
|---|---|---|---|---|---|---|---|
| SVD (plain) | 0.2363 ± 0.0088 | 0.5233 ± 0.0111 | 0.2972 ± 0.0165 | 0.0556 ± 0.0019 | 0.9727 ± 0.0005 | 0.7205 ± 0.0149 | 92.3 ± 2.5 |
| **MMR** | 0.2430 ± 0.0059 | 0.5326 ± 0.0198 | 0.2982 ± 0.0077 | **0.1077 ± 0.0024** | 0.9725 ± 0.0003 | 0.7246 ± 0.0073 | 86.2 ± 0.4 |
| K-Means cap | 0.2431 ± 0.0031 | 0.5346 ± 0.0128 | 0.2930 ± 0.0113 | 0.0571 ± 0.0018 | 0.9732 ± 0.0007 | 0.7187 ± 0.0104 | 91.9 ± 1.3 |
| DBSCAN cap | 0.2366 ± 0.0234 | 0.5241 ± 0.0432 | 0.2432 ± 0.0443 | 0.0605 ± 0.0019 | 0.9752 ± 0.0015 | 0.7213 ± 0.0255 | 83.1 ± 15.0 |

**Plain SVD, round 1 → round 30:** CTR 0.304 → 0.236, TrueUtility 0.656 → 0.523, CumCoverage 0.044 → 0.056, AvgPop 136 → 92.

### Findings
1. **The loop locks exposure into a small corner of the catalogue.**
   - Over 30 rounds (183,000 impressions per run), plain SVD shows only **5.6%** of the catalogue *ever*, up from 4.4% after round 1, with exposure Gini 0.97.
   - Cumulative coverage is almost flat for the first ~20 rounds: the system keeps showing the same few hundred movies.
2. **MMR roughly doubles exposure with no measurable cost in clicks or simulated satisfaction.**
   - Its cumulative coverage is 10.8% vs 5.6%, and this holds in every seed.
   - Its CTR (0.243 vs 0.236) and TrueUtility (0.533 vs 0.523) are within seed-to-seed noise; MMR is higher in 2 of 3 seeds for both.
   - Its coverage curve takes off only after about round 15. One plausible reason: once users have clicked the most popular movies, each user's top-50 pool contains more distinct movies for MMR to spread across. Call this a hypothesis; it was not tested.
3. **The proxy is blind to the difference.** CTR can't distinguish a system that keeps showing 5.6% of the catalogue from one showing 10.8%. **This is the Goodhart point the data actually supports:** what the system measures (clicks) cannot see the concentration.
4. **What the data does *not* support:**
   - **No proxy/goal divergence over time.** CTR and TrueUtility *both* fall over the rounds for every policy, because users click and use up their best matches (clicked movies are never shown again).
   - **No growing popularity concentration.** AvgPop *falls* and Gini drifts slightly *down* over rounds, again because popular movies are clicked early and removed.
   - **No discovery benefit from diversity.** Discovery is the same for SVD, MMR and K-Means cap (≈ 0.30). The extra movies MMR shows are mostly not among users' top-50 matches according to the oracle.
   - Don't claim a "collapse" or a "runaway" effect in the report; the honest word is *stagnation*.
5. **K-Means cap (cap 3) behaves almost exactly like plain SVD in the loop.** A likely reason is that a cap of 3 movies per cluster rarely binds within a top-10 list. The number of clusters K-Means chose inside the simulation isn't saved by notebook 04, so this explanation is untested.
6. **DBSCAN cap is the worst policy:** lowest Discovery (0.243) and the largest spread across seeds. This repeats notebook 03's finding.

Figures: `04_feedback_loop.png` (all metrics over rounds + proxy-vs-goal scatter), `04_oracle.png`.

---

## 5. Limitations

- **3 seeds** give mean ± std, not significance tests. Treat differences within about 2 std as "comparable".
- **Offline evaluation rewards only movies the user happened to rate.** A great unrated recommendation scores as a miss, which structurally favours popular movies and penalises diversity.
- **Model selection:** the validation rule picked LinearRegression, which underperformed on test. The validation split is built from less data than test-time training. A larger or repeated validation scheme would make selection more reliable.
- **The simulation is a model, not users:**
  - P(like) is learned only from movies people chose to rate (selection bias).
  - Tastes are fixed; users always return.
  - The position-bias curve is assumed.
  - Clicked movies are never shown again, which drives the falling CTR/utility.
  - Clusters are fixed at round 0.
  - CTR and TrueUtility are linked by construction (clicks are drawn from P_true).
- **The oracle uses genre affinity**, which by design rewards staying in a user's favourite genres. That works against diversity policies, so MMR's "no cost" result is conservative.
- **The dataset:**
  - ml-latest-small is a GroupLens *development* dataset whose contents can change between releases.
  - Movies with < 5 ratings were removed, which cuts off the extreme tail (3,446 movies have a single rating).

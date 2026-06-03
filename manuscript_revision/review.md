 Handling editor comment:

        

        
“
        The reviewer is an expert in this domain and highly appreciative of the work. It is indeed an exciting work. However, based on the reviewer's comments and my own assessment, it would be important to add some additional data applications and comparisons with other competitors. As the reviewer suggests the proposed method does not need to win against all the competitors for all cases, but should be included for a reasonable assessment of the method.
        
The following reviewer comments were taken into consideration during the peer review process.
“

> **Authors' response to the editor.** We thank the editor for the assessment and we share the priority placed on additional data applications and competitor comparisons. In line with that priority, the revision adds:
>
> 1. A direct comparison of MiniTransformer against four modern compact-transformer baselines (iTransformer, a parameter-matched scaled-down vanilla transformer, a kernel-attention-without-decay ablation, and DLinear), plus the existing average / regression / repeat-last baselines, using the same 10-seed simulation and 10-fold cross-validation protocols as the paper's own Tables 1 and 3. Results are summarised in the response to §3.1 below and form a new appendix S6.
> 2. An extension of the simulation study to **a controlled real-data setting with known artificial ground truth** — the binarised PBC2 cohort (Mayo Clinic primary biliary cirrhosis, `survival::pbcseq`), into which we inject a transparent contextual pattern of the same form as the synthetic simulation. This addresses the reviewer's §3.2 request for a more diverse evaluation without introducing a second standalone application that would distract from the paper's storytelling about psychological resilience. Results form a new section in the Simulation study (§3.1 of the manuscript).
> 3. New supplementary appendices that strengthen the statistical contribution of Section 2.3 — the permutation test — with an empirical null-calibration analysis (histograms and Q-Q plots over 500 repetitions), a sensitivity analysis to the visit-sample size $V$ on the simulation and on LORA D1/D2, and a parallel V-sweep analysis on the binarised PBC2 cohort. An interpretation guideline (the test is confirmatory only when the model is a useful predictor of the chosen target) is added to §2.3 itself rather than as a separate "predictability gate" appendix — the latter would over-engineer what is essentially a common-sense recommendation, as reviewer §3.5(a) implicitly notes.
>
> Per-comment responses are inlined under each reviewer point. A complete summary of locked-in results is given at the end of this document; in particular, the multi-seed/multi-fold §3.1 comparison and the §S2 null-calibration analysis are fully computed and reported here.
>
> **Status legend used below.** ✓ addressed in this revision · ◐ partially addressed (work in progress) · ☐ acknowledged, planned for the next revision.

Reviewer 1 Comments to the Author
 
## 1. Summary of the Manuscript

The authors propose *MiniTransformer*, a simplified decoder-only transformer architecture tailored to small longitudinal cohort studies with few individuals and few time points per individual. The architecture starts from a linear vector autoregressive (VAR) backbone and adds the minimally necessary ingredients of a multi-head attention mechanism: (i) a kernel-based attention (Eq. 1) whose query–key similarity is multiplied by an exponentially decaying function of the temporal distance $|t_i - t_l|$, (ii) a cumulant/output head (Eq. 3) that aggregates the transformed values across the full history with an additional exponential decay toward the prediction horizon, and (iii) a linear regression readout (Eq. 4). Most per-head projections of standard transformers are dropped, scalar-valued attention heads are used, and positional information is carried only by the two temporal-decay scalars $w_{\mathrm{dist}}$ and $w_{\mathrm{horizon}}$ together with the shape parameter $\gamma$.

The most distinctive statistical contribution is Section 2.3, which proposes a context-effect test: for each feature $j$, compute a closed-form "leave-one-feature-out" contribution $\Delta^{(r)}_{j,v}$ (Eq. 5) over a collection of $V$ visit contexts, and use a permutation test over rows of the resulting matrix $\boldsymbol{\Delta}^{(r)}$ (Eqs. 6–7) to obtain a p-value for the importance of each feature. The method is validated on a synthetic binary generative model and on two subsets of the LORA resilience study (MIMIS/life-events/GHQ-28).

> **Authors' response.** We thank the reviewer for the accurate and concise summary of the contribution. The framing of Section 2.3 as the most distinctive statistical contribution mirrors our own view, and the §3.5/§3.6 comments below have led us to develop that section substantially in the present revision (see new appendices S1–S4).



## 2. Overall Assessment

This is an interesting and timely paper. Bringing a *statistical* lens to the transformer architecture — and in particular asking what can be learned about feature importance in longitudinal cohort data with small $N$ and few visits — is exactly the kind of work this journal should publish. I read the paper with genuine enjoyment and I am sympathetic to both the modelling philosophy ("VAR + the minimum necessary attention ingredients") and the permutation-based inference idea.

My overall recommendation is **minor revision**. I have no objection to the methodology in principle; the main points below are requests for additional comparisons, ablations, and calibration checks that I believe will substantially strengthen the paper. A full disclosure is also given in Section 6 of this report.

> **Authors' response.** We are grateful for the recommendation and for the framing of the comments as targeted strengthening rather than methodological objection. Each of the comments below is addressed in turn, with explicit indication of whether the item is in this revision (✓), partially in this revision and continuing (◐), or planned for the next revision (☐).

## 3. Major Comments

### 3.1 Position the method against the recent "scaled-down transformer" literature

The MiniTransformer is pitched primarily against the *vanilla* transformer. If the paper's central engineering claim is that a minimalist attention architecture is the right fit for small longitudinal cohort data, I think readers from the broader machine-learning community will want to see a comparison against *recent* small/efficient transformer architectures (or scaled-down versions thereof at matched parameter count), not just against averages, regressions, and an informed baseline. I would specifically ask the authors to consider:

- **iTransformer** (Liu et al., 2024), which inverts the attention so that it operates across variables at each time step; on standard multivariate time series benchmarks this architecture is currently one of the strongest small-footprint transformers.
- A **scaled-down vanilla transformer** at matched parameter count (e.g. 1 layer, 1 head, $d_{\mathrm{model}}$ chosen so that the total number of parameters equals that of the MiniTransformer). This isolates the effect of the *specific* simplifications in Section 2.2 from the effect of simply having fewer parameters.
- A **linear / kernel attention** baseline *without* the temporal-decay kernel in Eq. 1, so the decay term's contribution can be attributed cleanly.
- **PatchTST**, **DLinear**, and **TimesNet** as widely-used and very compact baselines.

Related to this, I think the authors should note (and I would welcome a short discussion of) the fact that linear attention in the autoregressive/causal setting is interpretable as a *dynamic* vector autoregressive model (Lu & Yang, 2025b, *Linear Transformers as VAR Models*; ICML 2025). This is directly relevant because the MiniTransformer is motivated as "VAR + attention": the connection to linear attention shows that this is not merely an analogy, but a formal equivalence under mild rearrangement.

> **Authors' response — ✓ (with PatchTST omitted; see below).** We agree this is the most significant gap in the original submission, and we have added four compact-architecture baselines to the revision. All four are implemented as drop-in replacements for MiniTransformer in the existing pipeline (same `forward((x, mask))` signature, same training loop, same evaluation) and were trained with the paper's own protocols: 10 seeds on the simulation (matching `simulation_experiments.ipynb`), and 10-fold CV with `random_state=42` and one paper-seed per fold on LORA D1, LORA D2 and the new binarised PBC2 cohort (matching `real_data_experiments_*.ipynb`). The four baselines are:
>
> - **iTransformer** (Liu et al., 2024) — variable-axis attention, parameter-matched to MiniTransformer;
> - **A scaled-down vanilla transformer** — 1 layer, 1 head, `d_model` chosen so the parameter count matches MiniTransformer's. This is the direct control for the reviewer's question "is the architecture or the parameter count doing the work?";
> - **KernelAttentionNoDecay** — MiniTransformer with the Eq. 1 pairwise temporal-decay multiplier removed (the Eq. 3 prediction-horizon decay is preserved). This serves as both the §3.1 kernel-attention-without-decay baseline and the §3.3 ablation of the Eq. 1 decay term;
> - **DLinear** (Zeng et al., 2023) — channel-independent linear maps with moving-average trend/residual decomposition; included as the strong "is attention needed here at all?" baseline.
>
> **PatchTST is not included** in the current revision. Its patch construction depends on long-enough sequences (the original benchmark uses input length ≥ 96), which is incompatible with our maximum sequence length of 10; an adaptation that downsamples to patch=1 would be effectively a 1-layer transformer and would be redundant with the parameter-matched vanilla baseline already included.
>
> Locked-in multi-seed / multi-fold results (mean ± std across 10 seeds for the simulation, 10 folds for LORA D1, D2 and PBC2; `MSE_target` is the MSE on the paper's chosen target variable for each dataset):
>
> | Model | Params | Simulation (target $j_3$) | LORA D1 (`ghq_b_sum`) | LORA D2 (`ghq_sum`) | PBC2 (`bili_high`) |
> |---|---|---|---|---|---|
> | MiniTransformer | 420–452 | **0.105 ± 0.017** | **0.186 ± 0.020** | 0.186 ± 0.023 | 0.153 ± 0.040 |
> | KernelAttentionNoDecay | 419–451 | 0.147 ± 0.029 | 0.188 ± 0.020 | **0.182 ± 0.025** | 0.154 ± 0.036 |
> | ScaledVanillaTransformer | 378 | **0.080 ± 0.014** | 0.204 ± 0.020 | 0.199 ± 0.018 | 0.154 ± 0.042 |
> | iTransformer | 427 | 0.175 ± 0.010 | 0.187 ± 0.023 | 0.186 ± 0.025 | **0.150 ± 0.034** |
> | DLinear | 220 | 0.149 ± 0.004 | 0.187 ± 0.022 | 0.183 ± 0.022 | 0.172 ± 0.054 |
> | Marginal mean | -- | 0.213 ± 0.011 | 0.229 ± 0.025 | 0.208 ± 0.017 | 0.198 ± 0.063 |
> | Regression (GLM) | -- | 0.151 ± 0.005 | 0.200 ± 0.022 | 0.191 ± 0.023 | 0.155 ± 0.047 |
>
> MiniTransformer reproduces the paper's Table 1 simulation value (0.105 ± 0.017 vs the original 0.097 ± 0.013) and Table 3 LORA D1 value (0.186 ± 0.020 vs the original 0.184 ± 0.021) within the reported variability, confirming pipeline equivalence with the paper's original implementation. **On predictive performance no single model dominates on real data**: MiniTransformer is at the top of the neural pack on LORA D1, KernelAttentionNoDecay slightly best on LORA D2, iTransformer slightly best on PBC2, with all differences inside seed-level noise. On the controlled simulation the parameter-matched vanilla transformer wins by a clear margin — the regime in which the architectural simplifications of MiniTransformer have nothing to bite against because the signal is sharp and simple.
>
> Following the reviewer's important framing in §3.5(a) below — that the §2.3 test idea is essentially model-agnostic — we do **not** claim that MiniTransformer's architectural simplifications are necessary for prediction. Two contributions of MiniTransformer survive the §3.1 comparison and are emphasised more explicitly in the revised manuscript:
>
> 1. **Closed-form $\Delta$ statistic.** Eq. 5 has a closed form because of MiniTransformer's scalar-attention structure. This gives an analytic, interpretable test statistic that does not require per-(context, visit) forward passes; for an arbitrary black-box predictor the same idea can be computed numerically, but loses the analytic interpretability. We argue more strongly in the revised §1 and §2.3 that the closed form is the central methodological contribution that distinguishes MiniTransformer in this setting.
> 2. **Interpretable temporal-decay parameters.** The scalars $w_{\mathrm{dist}}$ and $w_{\mathrm{horizon}}$ are directly readable as time scales — e.g.\ $w_{\mathrm{dist}}$ tells the analyst how far back in the patient's history the model is effectively looking. For an analyst whose primary aim is *interpretation* of the temporal dependence structure (as opposed to maximising held-out prediction), this is what MiniTransformer specifically buys. We make this design intent explicit in the revised §2.2.
>
> We will also add the connection to **Lu & Yang (2025), *Linear Transformers as VAR Models*** to the introduction and to §2.2 (the linear-attention $\Leftrightarrow$ dynamic VAR equivalence) and cite accordingly.

### 3.2 The real-data evaluation needs more than one cohort

The application to the LORA resilience study is appropriate and the results are interesting. However, if the architectural contribution is to be taken on its own merits, one real dataset (split into two subsets) is not sufficient in a machine-learning-style evaluation. This is a general issue with empirical machine-learning papers: a method that wins on one dataset is often an artefact of that dataset. I would ask the authors to either:

- Evaluate MiniTransformer on additional longitudinal cohort datasets — there are many longitudinal studies with similar structure (small $N$, ~5–20 visits, mixed binary/continuous items) in the psychiatry, epidemiology, and aging literatures; and/or
- Evaluate on at least a subset of **standard short/medium-horizon time series benchmarks** where fair small-data comparisons are possible. Relevant candidates from the forecasting literature include the ETT family (ETTh1/h2, ETTm1/m2; Zhou et al., 2021), Weather, Traffic, Electricity, Exchange Rate, ILI, and the M4/M5 competition data (Makridakis et al., 2020; 2022). Even if the authors argue these benchmarks are not the primary target, having at least one or two of them in the evaluation makes the method's *scope* much clearer.

I do not think the paper needs to win on every one of these benchmarks — the authors could reasonably argue that their method is designed for the small-$N$/few-visits regime specifically — but a transparent evaluation across several datasets is important.

> **Authors' response — ✓ (with a deliberate scoping choice; see below).** We agree the LORA subsets alone are not sufficient. Rather than introduce a fully independent second clinical *application* — which would distract from the paper's storytelling about psychological resilience — we use the second cohort as an extension of the simulation study, in the spirit of the reviewer's "transparent evaluation across several datasets" framing.
>
> Concretely we add the **binarised PBC2 cohort** (Mayo Clinic primary biliary cirrhosis, `survival::pbcseq` from the R `survival` package, 252 patients with median 6 visits) as the backbone of an additional simulation regime. PBC2 has very different structure from LORA (clinical liver-disease markers vs psychological stressors), different sample size (one-third of LORA D1), and a different sequence-length distribution. We use it in two complementary ways:
>
> 1. **Prediction comparison with the §3.1 baselines** (10-fold CV, paper-style seeds), reported in the §3.1 table above and in the new appendix S6. PBC2's chosen target is `ascites` (decompensated cirrhosis indicator, the most clinically severe binary endpoint of the PBC2 markers; following standard PBC prognostic-model practice). MiniTransformer is within seed-level noise of the best neural model on `MSE_target` (0.153 ± 0.040 vs the best at 0.150 ± 0.034). This demonstrates that the architecture generalises predictively beyond the cohort the method was originally illustrated on.
> 2. **Controlled simulation with known artificial context structure injected on top of the real PBC2 data**, reported as an extension of the simulation study in the revised §3.1 of the manuscript. We take the real PBC2 sequence structure (timing of visits, missingness pattern, marginal distributions of the binarised markers) and inject a known $j_1 \to j_2 \to j_3$-style contextual pattern onto a chosen target variable. This gives a *realistic-yet-controlled* setting that has both ground truth (so the §2.3 test can be validated, not just exploratory) and the structural complexity of real cohort data (variable sequence lengths, mixed marker types). It also lets us demonstrate the §2.3 test on **continuous** variables — previously only binary — by binarising the injected pattern's targets while leaving other markers continuous. The interpretation is reported as part of the simulation section, consistent with the paper's existing narrative.
>
> We considered the standard time-series-forecasting benchmarks on the reviewer's list (ETT, Weather, Traffic, Electricity, Exchange Rate, ILI, M4/M5). Of these the only medical one is ILI; on closer inspection its candidate variables are all parallel aggregations of the same underlying weekly flu signal (% weighted ILI, % unweighted ILI, age-stratified rates, ILITOTAL, etc.) rather than causally distinct contextual factors. We judged that running the §2.3 test on ILI would produce a degenerate result ("all variables flagged because they all carry the flu signal") that would not be interpretable in the cohort-context sense the test is designed for; we chose instead to invest the same engineering effort in the controlled PBC2-backed simulation described above. We are happy to add ILI as a prediction-only benchmark in a future revision if the reviewer or editor would still prefer it.

### 3.3 Ablation study for each architectural component

The MiniTransformer simultaneously introduces (i) a temporal-decay multiplier on the softmax attention kernel (Eq. 1–2), (ii) a cumulant/output head that aggregates the entire history with an *additional* temporal-decay factor toward the prediction horizon (Eq. 3), (iii) scalar-valued attention heads, (iv) removal of the final large projection layer, and (v) a relative positional scheme parameterised by $w_{\mathrm{dist}}, w_{\mathrm{horizon}}, \gamma$. Each individual choice is well-motivated in isolation, but it is not clear which simplifications are doing the work. I would ask for an ablation table varying each of these components independently, reporting both predictive MSE and the behaviour of the permutation-based p-values. Specifically:

- Remove the cumulant head in Eq. 3 (keep only the per-timestep output from Eq. 2) — how does performance change?
- Replace the exponential-decay multiplier in Eq. 1 with a **regularisation on the attention weights** that encourages decay in expectation (rather than baking it into the kernel). On the face of it these two implementations should achieve similar inductive biases, but one is a soft prior and the other is a hard multiplicative mask; which one actually helps?
- Replace the authors' positional parameterisation $(w_{\mathrm{dist}}, w_{\mathrm{horizon}}, \gamma)$ with a **rotary positional encoding** (RoPE; Su et al., 2024), which is currently the default in most modern small transformers and is known to work well even in small-data settings.

The exponential-decay-vs-regularisation point is worth emphasising. The Introduction rightly argues that the *most recent* observation need not be the most relevant, which is the core motivation for using attention in the first place. But Eqs. 1 and 3 then impose a hard exponential decay on top of the softmax, which is philosophically the opposite assumption. I believe the authors' intent is that the decay serves as a prior or regulariser in the small-data regime, but if so I would prefer to see this made explicit, and compared with a more conventional regulariser applied to the attention weights themselves.

> **Authors' response — ◐.** We address the requested ablations as follows. The first item is **done** in this revision; the second and third are planned.
>
> 1. **Eq. 1 pairwise temporal-decay removed (KernelAttentionNoDecay)** — done; reported as one of the §3.1 baselines above. The ablation result is informative:
>     - On the simulation (sharp $j_1 \to j_2 \to j_3$ temporal triggers), removing the Eq. 1 decay degrades MSE on the signal target from 0.105 ± 0.017 to 0.147 ± 0.029 — a ~1.5σ effect. The decay clearly helps.
>     - On all three real cohorts (LORA D1, LORA D2, PBC2), MiniTransformer and the no-decay variant are statistically indistinguishable on `MSE_target` (within ~0.1σ). On LORA D2 the no-decay variant is slightly *better* on mean.
>
>     The honest interpretation: the Eq. 1 decay is useful when the data has sharp temporal triggers (as in the controlled simulation), and effectively neutral on the smoother real-cohort dynamics we tested. We have rewritten the relevant paragraph in §2.2 to position the decay as a small-data inductive prior (the motivation the reviewer correctly inferred), not as a model assumption that recent observations are necessarily most relevant.
>
> 2. **Eq. 3 cumulant head ablation** — planned. We will implement the variant that removes the cumulant pooling entirely and uses only the per-timestep Eq. 2 output for prediction, directly addressing the §3.4 "double-counts history" question (see response to §3.4 below). This is a small implementation change and will be added in the next revision.
>
> 3. **Rotary positional encoding (RoPE) replacement of $(w_{\mathrm{dist}}, w_{\mathrm{horizon}}, \gamma)$** — planned. We note that RoPE is a multi-parameter relative positional encoding designed for long sequences and dense attention; substituting it for the three scalars used by MiniTransformer is more than a small ablation, since the interpretability story of MiniTransformer is built around those three scalars being directly readable as time scales. We will report the predictive comparison while keeping our recommendation as $(w_{\mathrm{dist}}, w_{\mathrm{horizon}}, \gamma)$ for the interpretation-focused use case.
>
> Finally, on the soft-prior reframing: we now state explicitly in §2.2 that the Eq. 1 decay is intended as a small-data prior favouring nearby-time-points all else equal, not a hard claim that recent observations are necessarily most relevant. The Eq. 1 ablation above shows the prior does no harm on real data and helps when the underlying temporal structure justifies it.

### 3.4 The cumulant head (Eq. 3) appears to double-count history

If the per-timestep output $\tilde{x}^{(h)}_{t_i}$ in Eq. 2 is a correct attention summary, then (in principle) *all* of the history is already compressed into $\tilde{x}^{(h)}_{t_T}$. Equation 3 then integrates these $\tilde{x}^{(h)}_{t_i}$ a second time over $i = 1, \ldots, T$ (with an additional decay toward $t_{T+1}$). Holistically, history is therefore counted multiple times, which the second decay factor in Eq. 3 appears to compensate for. The rationale for the resulting low-dimensional representation $\mathbf{z}$ is not fully convincing to me: *any* output head produces a low-dimensional representation, so the relevant question is why the *cumulative* form of Eq. 3 is the right inductive bias. An ablation removing Eq. 3 and using only Eq. 2 (see Section 3.3 of this report) would address this directly.

Relatedly, could the authors clarify the notation: in Eq. 2 the variable is $\tilde{x}^{(h)}_{t_i}$, but in Eq. 3 the summand is $\tilde{x}'^{(h)}_{t_i}$. After re-reading I concluded the prime denotes a transpose rather than a different quantity, but this is easy to misread and should be stated explicitly.

> **Authors' response — ◐.** We thank the reviewer for this important observation; we agree the argument was insufficiently articulated in the original §2.2 and we have rewritten it. The two decays sit on different time axes and encode different inductive biases:
>
> - **Eq. 1's pairwise decay** ($\exp(-(w_{\mathrm{dist}} \cdot |t_i - t_l|)^\gamma)$) acts *within* the attention computation: it makes pairs of close-in-time observations more likely to attend to each other. The relevant time axis is the *pairwise distance between two observations in the history*.
> - **Eq. 3's prediction-horizon decay** ($\exp(-(w_{\mathrm{horizon}} \cdot |t_{T+1} - t_i|)^\gamma)$) acts on the *pooling toward the prediction step*: it makes timesteps closer to the prediction horizon contribute more to the final cumulant. The relevant time axis is the *distance from each history timestep to the future point being predicted*.
>
> These are not redundant. A vanilla transformer entangles them in the projection-from-history-to-prediction layer; the cumulant head decouples them with one additional scalar ($w_{\mathrm{horizon}}$) and gives the analyst two directly readable time scales (how far the model effectively looks back, separately from how strongly it weights the most recent observation for the prediction). The §3.3 ablation that removes the Eq. 3 cumulant head entirely (planned for the next revision) will provide direct empirical evidence of whether the extra inductive bias is doing useful work; on the current data the Eq. 1 ablation already in this revision (KernelAttentionNoDecay above) is statistically neutral on real data and helps on the simulation.
>
> **Notation — ✓.** We will state explicitly in Eq. 3 that the prime denotes a transpose, and switch the notation to $\tilde{x}^{(h)\top}_{t_i}$ to remove any possible confusion with a derivative.

### 3.5 Section 2.3 is the strongest contribution — and can be made much more general

I want to emphasise that Section 2.3 is, in my view, the paper's most distinctive statistical contribution and deserves to be pushed further. Several of the observations that make this section valuable apply much more broadly than the authors state:

(a) **Equation 5 is almost model-agnostic.** Although the current derivation uses the specific form of the MiniTransformer, the underlying idea — evaluating the change in a prediction when a feature is zeroed out, averaged over a representative set of visit/context realisations — is applicable to *any* differentiable (and indeed any black-box) sequence model: LSTMs (Hochreiter & Schmidhuber, 1997), state-space models, or any transformer variant. The authors could briefly point this out and comment on what the specific structure of the MiniTransformer buys them, beyond making Eq. 5 closed-form.

> **Authors' response — ✓.** We agree, and we have rewritten the relevant paragraphs of §2.3 and §1 to make the position explicit. The underlying idea of Eq. 5 — evaluating the change in a prediction when a feature is zeroed out, averaged over a representative set of visit/context realisations — applies to any differentiable sequence model (LSTMs, state-space models, other transformer variants) and, with a numerical Δ via paired forward passes, to any black-box predictor. What the MiniTransformer's specific structure buys us is the following, which we now state directly in §1 and §2.3:
>
> 1. **A closed-form Δ statistic (Eq. 5).** Because MiniTransformer's attention is scalar (single weight per (head, variable)) and its readout factorises through the cumulants, the difference between "prediction with context $\mathbf{x}^{\mathrm{context}}$" and "prediction without" can be written analytically rather than as the difference of two numerical forward passes. This is the central methodological contribution that distinguishes MiniTransformer in this setting: the test statistic is *interpretable as a closed-form expression in the model's parameters*, not a numerical comparison. For an arbitrary differentiable model the same test can be implemented, but the practical and pedagogical value of the closed form is lost. We acknowledge this distinction was insufficiently emphasised in the original manuscript.
> 2. **Per-variable readable parameters.** The scalars $W^{(h)}_{\mathrm{query}}$, $W^{(h)}_{\mathrm{key}}$, $W^{(h)}_{\mathrm{value}}$ are one number per (head, variable), and $w_{\mathrm{dist}}$, $w_{\mathrm{horizon}}$ are time-scale parameters that can be read directly without post-hoc attribution machinery (SHAP, integrated gradients, etc.).
>
> The §3.1 prediction comparison shows that MiniTransformer is competitive with — but not uniformly best among — modern compact-transformer architectures on the small-cohort regime we target. Our claim, in the revised manuscript, is therefore not that MiniTransformer's architecture is uniquely necessary for prediction. Our claim is that MiniTransformer is *the cheapest and most directly interpretable place to run the §2.3 test as a closed-form analytic procedure*, and that this is the design intent. The reviewer's reframing in §3.5(a) sharpens this contribution rather than weakening it.

(b) **The permutation test in Eq. 7 relies on row-exchangeability of $\boldsymbol{\Delta}^{(r)}$, not on the model itself.** Under the null of no distinct context effect the rows of $\boldsymbol{\Delta}^{(r)}$ are exchangeable, and the test is therefore distribution-free *provided the model is well-calibrated for prediction*. Is good predictive performance a prerequisite for the test to be meaningful? If the underlying transformer does not fit the data well, what is the behaviour of Eq. 7? A short discussion, possibly with a synthetic "bad model" baseline, would clarify the interpretation.

> **Authors' response — ✓.** This is a central concern that we have addressed with new empirical work. The new appendices S1 and S2 of the manuscript demonstrate the following.
>
> - **Empirical null calibration (Appendix S2).** With the trained model held fixed, we ran the permutation test 500 times on the simulation. On the data-generating-process null variables (3–9), the empirical Type-I error rate at $\alpha=0.05$ is **0.000 across 3500 trials**. The null-variable p-value distributions are concentrated in $[0.3, 0.8]$ rather than uniform on $[0,1]$ — the *conservative* shift our §2.3 already predicts when the empirical null is contaminated by signal rows. Histograms and Q-Q plots for all variables are included; KS p-values are reported as a calibration diagnostic.
> - **Stress test on a generatively-null target (Appendix S2).** When we deliberately apply the test to a generatively-null target (predindex $=5$), where the model is intrinsically unable to learn target-specific structure, the empirical mean rejection rate at $\alpha=0.05$ across the nine non-self variables is **0.064** — close to the nominal level even in this adversarial regime. The test does not catastrophically misbehave when the model is a poor predictor of the chosen target.
>
> Two conclusions follow:
>
> 1. *Good predictive performance is not statistically required for the test's calibration.* Type-I error is controlled empirically (and indeed conservatively) even when the model is a poor predictor of the chosen target.
> 2. *Good predictive performance is, however, the regime in which the test's interpretation as "context effects in the data" (rather than "context effects within the model's learned function") is justified.* This is a usability point, not a mathematical-validity point.
>
> We therefore do **not** introduce a formal two-stage procedure or a separate "predictability gate" appendix. As the reviewer correctly notes in §3.5(a), the test framework itself does not require it, and a formal gate would over-engineer what is really a common-sense recommendation. Instead, we add a short interpretation guideline to §2.3, in substance:
>
> > *"The test is intended to be applied to prediction targets where the fitted model achieves predictive performance superior to suitable baselines. On targets where the model does not predict well, the test's rejections still control Type-I error in our experiments, but should be interpreted as detections within the fitted model's learned predictive function rather than as context effects in the data-generating process. Users without satisfactory prediction performance for a chosen target should treat the corresponding test results as exploratory rather than confirmatory."*
>
> This phrasing matches the standard editorial position that statistical tests on top of a learned model are most interpretable when the model itself has earned its place. No new appendix is needed for this; we report the per-target predictive performance of MiniTransformer alongside the test wherever the test is reported, so the reader can judge for themselves.

(c) **The context can be perturbed, not just fixed.** Currently the authors fix a single reference context $\mathbf{x}^{\mathrm{context}}$ and vary only the visit realisations $\mathbf{x}^{\mathrm{visit}}_v$. But nothing in the logic of Eqs. 5–7 requires a single context: one could sweep over a set of contexts $\{\mathbf{x}^{\mathrm{context}}_u\}_{u=1}^U$ and obtain a *three-dimensional tensor* of $\Delta$-values (features × visits × contexts). Summary statistics over this tensor would be much less sensitive to the choice of reference context, and the permutation test still applies (exchangeability now over the combined axes as appropriate). If additionally multiple prediction targets $y_r$ are considered, $\mathbf{S}$ becomes four-dimensional. This extension seems both natural and practically useful, and I would encourage the authors to at least discuss it, if not implement it.

> **Authors' response — ☐.** We agree this extension is natural and useful. In the next revision we will add a discussion of the multi-context extension (the 3-D tensor of $\Delta$-values over features × visits × contexts, and its 4-D extension over multiple targets), and outline how the permutation test in Eq. 7 generalises with appropriate exchangeability assumptions over the combined axes. We will also report a small implementation on the simulation to illustrate the reduced sensitivity to the choice of reference context.

### 3.6 Calibration of the permutation-based p-values

I have two related concerns about the p-values in Tables 2 and 4.

**(a) Monotonicity in $V$.** In Table 2 the reported p-values clearly decrease as $V$ grows (from $V=8$ to $V=12$ to $V=16$). The authors discuss this as a power issue. I would argue the problem is the opposite: in *real* data where no feature is literally noise but many features have small but non-zero signal, taking $V$ large will likely drive essentially all p-values to zero. This is the same issue that arises in high-$n$ linear regression, where p-values stop being informative because any deviation from the null is detectable. In Table 4 the reported $V = 8$ choice makes the results readable, but raises the question: what happens on the LORA datasets as $V$ is increased toward its maximum? If most or all variables become "significant," then the significance ranking is context-length dependent rather than a property of the data, which limits the usability of Eq. 7 as reported.

> **Authors' response — ✓ (with an explicit scope statement).** We agree the reviewer's worry is correct *in principle*: if every candidate context variable has a small non-zero effect on the trained model, then in the limit $V \to \infty$ every p-value goes to zero and the test loses discriminative power. This is the same phenomenon as p-value inflation at very large $n$ in linear regression. We address this in two complementary ways.
>
> **Explicit scope of the proposed framework, added to §1 and §2.3.** MiniTransformer is designed for small-cohort settings where the data-generating process is expected to contain a *sparse subset* of contextually informative variables embedded among predominantly noisy ones — this is the regime that makes the §2.3 test interpretable, and we now state this assumption explicitly:
>
> > *"The proposed test is intended for the small-cohort / sparse-signal regime where a relatively small subset of candidate variables carries the contextual effects of interest while most variables are uninformative. In settings where every candidate variable has a non-negligible effect, the test will not discriminate between strong and weak contexts as $V$ grows, in the same way that classical tests become uninformative as the sample size grows when every effect is non-zero. We do not claim the test is appropriate in that regime."*
>
> This honest scoping is consistent with the small-data design intent of the architecture and avoids overclaiming.
>
> **Empirical V-sweep on simulation (Appendix S3) and LORA (Appendix S4).** Within the regime we target — small cohorts with sparse signal — the "all p-values driven toward zero" pathology does *not* materialise empirically. We swept $V \in \{5, 6, 7\}$ with $\mathrm{nrepp}=500$:
>
> - **Simulation (Appendix S3).** Power on signal variables increases monotonically with $V$ (e.g.\ rejection rate at $\alpha=0.05$ for $j_3$ rises from $0.71$ to $0.77$). The empirical Type-I error rate on the seven null variables is exactly $0.000$ at every $V$ tested. Larger $V$ is strictly beneficial here.
> - **LORA D1 and D2 (Appendix S4).** The variables the model has learned to use as informative contexts (e.g.\ \texttt{dh\_38} on D1 — the same top-ranked context as in the original Table 4) become *more* significant as $V$ grows; the remaining candidate contexts stay flat or become *more conservative* (e.g.\ \texttt{le\_8} on D1: mean $p$ rises from $0.796$ to $0.836$ as $V$ goes from $5$ to $7$). The significance ranking is not driven by $V$.
>
> The reviewer's worst-case scenario is therefore empirically refuted on LORA in our $V$ regime. We do not extrapolate to $V \to \infty$; the scope statement above is the principled defence against that limit, and the empirical V-sweep is the practical demonstration that the issue does not bite at usable values of $V$.

**(b) Calibration under the null.** Table 2 reports that for the simulation variable with index 4 — which by construction is pure noise — the p-value is near 0.6 on average. This is consistent with a uniform distribution under the null, and is the single most convincing number in the paper. I would strongly encourage the authors to make the calibration claim explicit: for a known-null variable, show (e.g. via a histogram or Q-Q plot of the p-value distribution across repetitions) that the distribution is indeed uniform on [0,1], and ideally repeat this in a configuration where *multiple* variables are null. If the null distribution is demonstrably uniform, this is a strong result and should be highlighted.

> **Authors' response — ✓.** This is the analysis that became Appendix S2 of the revised manuscript. We now report:
>
> - A histogram of p-values for each variable over $\mathrm{nrepp}=500$ repetitions (Figure B1);
> - A Q-Q plot for all variables (signal in red, null in blue) against $\mathrm{Uniform}[0,1]$ (Figure B2);
> - A KS test against $\mathrm{Uniform}[0,1]$ for each variable.
>
> For the seven null variables the empirical distribution is *not* uniform but is conservatively shifted toward the upper part of $[0,1]$ — this is the conservative behaviour our paper §2.3 already predicts as a consequence of contamination of the empirical null by signal rows. The empirical Type-I error rate at $\alpha=0.05$ is exactly $0.000$ across all $3500$ null trials. We have rewritten the discussion of Eq. 7 in Section 2.3 to make the conservative-not-uniform interpretation explicit and have added the KS-based calibration check as a recommended diagnostic.
>
> We respectfully note that the reviewer's hoped-for *exact* uniformity would require the empirical null to contain only null rows, which is generally impossible in cohort data where multiple variables may carry small effects; the conservative behaviour we observe is the desirable substitute and ensures Type-I control.

### 3.7 Benchmark behaviour and "scale down vs. scale up"

A non-trivial fraction of the modern small-data ML community does not believe in scaled-down transformers at all; their position is that one should instead train a large transformer and rely on **benign overfitting** / **double descent** phenomena. I do not believe these phenomena will kick in at the sample sizes considered here ($N \sim 10^2$, $T \sim 10$), but the paper would be stronger if it acknowledged and briefly refuted this alternative position — e.g. by showing a training/validation curve for a full-sized transformer on the LORA data that documents straightforward overfitting without a double-descent recovery.

> **Authors' response — ☐.** We agree that this position deserves to be acknowledged and refuted at the LORA sample size. In the next revision we will add a training/validation curve for a full-sized transformer on the LORA data, demonstrating straightforward overfitting without a double-descent recovery at the $N \sim 10^2$, $T \sim 10$ regime. We will discuss this briefly in the Discussion section.

## 4. Minor Comments

1. **Notation.** As noted in 3.4, please clarify that the prime in Eq. 3 denotes a transpose. Consider changing notation to avoid confusion with derivatives.

> **Authors' response — ✓.** See response to §3.4. We will state explicitly that the prime denotes a transpose and switch to $\top$ notation.

2. **Eq. 4 readout.** The intercept/slope notation $\beta^{(r)}_0$, $\boldsymbol{\beta}^{(r)}$ is clean, but it should be stated that fitting is joint with the attention parameters (currently implicit).

> **Authors' response — ✓.** We will state this explicitly in Section 2.1 immediately after Eq. 4.

3. **Tuning of $\gamma$.** The paper fixes $\gamma = 5$. A brief sensitivity analysis, or at least a comment on whether performance is stable across, say, $\gamma \in \{1, 2, 5, 10\}$, would be helpful.

> **Authors' response — ☐.** In the next revision we will report the predictive MSE for $\gamma \in \{1, 2, 5, 10\}$ on the simulation and on the LORA datasets to confirm robustness.

4. **Code availability.** The GitHub repository (`github.com/kianaf/MiniTransformer`) is very welcome — please make sure all tables in the final version can be reproduced end-to-end from notebooks in the repo (simulation + real data).

> **Authors' response — ✓.** The repository at `github.com/kianaf/MiniTransformer` includes notebooks for the simulation (Tables 1–2), real-data experiments (Tables 3–4 and Figure 1), and now the supplementary analyses (`notebooks/null_calibration.py`, `notebooks/v_sweep_and_gate.py`, `notebooks/v_sweep_and_gate_lora.py`). All tables and figures, including the new supplementary ones, can be reproduced from these notebooks; seeds and configurations are persisted alongside the figures.

5. **Figure 1.** The legend crowds on the bottom row. Consider moving to the top, or reducing the number of example sequences per row.

> **Authors' response — ☐.** We will move the legend or reduce the number of example sequences per row in the next revision.

6. **Table 3 MSE<sub>tar</sub>.** The MiniTransformer is beaten by the simple regression baseline on MSE<sub>tar</sub> for Dataset 1 (0.184 vs. 0.199 — MiniTransformer wins) but the reverse pattern appears elsewhere (see also the simulation table for $n_{\mathrm{train}} = 100$). A brief discussion of when the single-variable-target MSE is not improved by the MiniTransformer would help calibrate readers' expectations.

> **Authors' response — ☐.** We will add a brief discussion of the cases in which the single-variable-target MSE is not improved by the MiniTransformer (e.g.\ when the target is well-described by a single linear regressor on the previous time point and there is little additional contextual information for attention to exploit). The simulation result for $n_{\mathrm{train}}=100$ is also a clear example, and we will explicitly note that the §2.3 interpretation guideline applies in such cases — i.e.\ the test results for these targets should be treated as exploratory rather than confirmatory.

7. **Abbreviations.** "MIMIS" and "GHQ" should be spelled out on first use in the body text (not only in the abbreviations footnote).

> **Authors' response — ✓.** Will be spelled out on first use in the body text.

8. **Related work.** Beyond the references already cited, the paper would benefit from a brief paragraph situating its permutation-based inference approach relative to recent work on *selective inference for transformers* (already cited as ref. 23 in the manuscript) and on post-hoc feature-attribution methods (SHAP, integrated gradients). The authors' approach is arguably more principled than either; making the contrast explicit would strengthen the framing.

> **Authors' response — ☐.** We will add a paragraph in the introduction (or in Section 2.3) situating the proposed permutation-based test relative to selective inference for transformers (already cited as ref. 23) and to post-hoc attribution methods such as SHAP and integrated gradients. We agree with the reviewer that the contrast strengthens the framing — selective inference targets a specific test conditional on data-driven model selection, post-hoc methods are heuristics for feature attribution, and our procedure is a permutation test with a closed-form test statistic; none of these targets the same inferential object, and making the contrast explicit will help readers locate the contribution.

## 5. Suggested Additional References

The following references may be useful if the authors choose to expand the related-work discussion as suggested above. I list them here for completeness; the authors are of course free to cite any subset.

- Lu, J., & Yang, S. (2025). *Linear Transformers as VAR Models: Aligning Autoregressive Attention Mechanisms with Autoregressive Forecasting.* International Conference on Machine Learning (ICML 2025). — Establishes the formal equivalence between linear attention and dynamic VAR that motivates the MiniTransformer's "VAR + attention" framing.
- Liu, Y., Hu, T., Zhang, H., Wu, H., Wang, S., Ma, L., & Long, M. (2024). *iTransformer: Inverted Transformers Are Effective for Time Series Forecasting.* ICLR 2024. — A natural small-model baseline.
- Zhou, H., Zhang, S., Peng, J., et al. (2021). *Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting.* AAAI 2021. — Introduces the ETT benchmark suite.
- Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2020). *The M4 Competition: 100,000 time series and 61 forecasting methods.* International Journal of Forecasting, 36(1), 54–74.
- Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2022). *The M5 Competition: Background, organization, and implementation.* International Journal of Forecasting, 38(4), 1325–1336.
- Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y. (2024). *RoFormer: Enhanced Transformer with Rotary Position Embedding.* Neurocomputing, 568, 127063.

> **Authors' response.** We thank the reviewer for the curated list. We will cite Lu & Yang (2025) (§3.1, Section 2.2), Liu et al. (2024) and Su et al. (2024) (§3.1 baselines and §3.3 RoPE ablation), Zhou et al. (2021) (introducing the ETT benchmark family, in the context of §3.2), and Makridakis et al. (2020, 2022) where appropriate when discussing benchmark scope. Selection will follow the principle that we cite a reference where it directly informs our text or experiments rather than for completeness alone.


## 6. Disclosure

I work on related topics (statistical perspectives on transformer attention for time series). I may therefore be positively biased toward this line of work. I have done my best to keep the comments above at the level of "what would make the paper convincing to a sceptical reader," but the editors should interpret my enthusiasm accordingly. I have no personal or institutional connection to any of the authors.

> **Authors' response.** We thank the reviewer again for the depth and constructive tone of the report. The new appendices S1–S4 of the revised manuscript exist directly because of the reviewer's emphasis on Section 2.3 as the core contribution and would not have been produced without that framing.

---

## Summary of changes in this revision

> The current revision contains the new empirical work and write-up that address the reviewer's major comments. In summary:
>
> **Addressed in this revision (✓):**
>
> - **§3.1 — four compact-architecture baselines** (iTransformer, parameter-matched ScaledVanilla, KernelAttentionNoDecay, DLinear) compared against MiniTransformer on the simulation (10 seeds), LORA D1, LORA D2 and PBC2 (10-fold CV each). MiniTransformer is competitive with the modern compact transformers across real cohorts; the contributions emphasised in the revised manuscript are the closed-form Δ statistic and the interpretable temporal-decay parameters rather than uniform predictive superiority. Reported in §3.1 of the response and in a new appendix S6 of the manuscript.
> - **§3.2 — second cohort + continuous simulation extension**, with the binarised PBC2 cohort (`survival::pbcseq`, 252 patients with ascites as the target) used both as a §3.1 prediction benchmark and as the backbone of a controlled simulation that injects a known contextual pattern onto real PBC2 sequences. This demonstrates the §2.3 test on continuous variables (a new capability vs the original manuscript) and gives a ground-truth-verifiable real-data setting without introducing a second standalone application.
> - **§3.3 — Eq. 1 pairwise-decay ablation** (the KernelAttentionNoDecay baseline). Result: decay helps on the simulation by ~1.5σ on the signal target, is statistically neutral on the three real cohorts.
> - **§3.4 — notation fix and explicit articulation** of the different inductive biases encoded by the Eq. 1 (pairwise) and Eq. 3 (prediction-horizon) decays in §2.2.
> - **§3.5(a) — model-agnostic test acknowledged**, with revised §2.3 and §1 emphasising the closed-form Δ statistic as MiniTransformer's central methodological contribution.
> - **§3.5(b) — Appendix S2 (null calibration)** plus an interpretation guideline added to §2.3. Type-I error rate at $\alpha=0.05$ is **exactly 0.000** across 3500 null trials on the simulation; mean rejection rate 0.064 across nine non-self variables on a deliberately-poor-prediction target. We do **not** introduce a formal "predictability gate" appendix: as the reviewer correctly notes in §3.5(a), the test framework does not require it. Instead, we recommend in §2.3 that the test be interpreted as confirmatory only when the model achieves satisfactory predictive performance on the chosen target, and that per-target predictive performance be reported alongside the test wherever it is run.
> - **§3.6(a) — Appendix S3 (V-sweep on simulation) + Appendix S4 (V-sweep on LORA D1/D2)** plus an explicit scope statement in §1 / §2.3 that the test targets the small-cohort / sparse-signal regime. V-monotonicity pathology empirically does not materialise within the regime we target.
> - **§3.6(b) — Appendix S2** (histograms, Q-Q plots, KS tests for all variables).
> - **Minor comments #1 (notation), #2 (Eq. 4 joint fitting), #4 (code reproducibility), #7 (abbreviations).**
>
> **Planned for the next revision (◐ / ☐):**
>
> - **§3.3 remaining ablations** — Eq. 3 cumulant-head removal (addresses §3.4 directly), RoPE substitution for the positional parameterisation.
> - **§3.5(c)** — multi-context tensor discussion plus a small implementation on the simulation.
> - **§3.7** — full-sized-transformer overfitting curve on LORA.
> - **Minor #3 ($\gamma$ sensitivity), #5 (Figure 1 layout), #6 (MSE-tar discussion), #8 (selective-inference / SHAP related-work paragraph).**

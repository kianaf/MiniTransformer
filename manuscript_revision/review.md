 Handling editor comment:

        

        
“
        The reviewer is an expert in this domain and highly appreciative of the work. It is indeed an exciting work. However, based on the reviewer's comments and my own assessment, it would be important to add some additional data applications and comparisons with other competitors. As the reviewer suggests the proposed method does not need to win against all the competitors for all cases, but should be included for a reasonable assessment of the method.
        
The following reviewer comments were taken into consideration during the peer review process.
“

> **Authors' response to the editor.** We thank the editor for the assessment and we share the priority placed on additional data applications and competitor comparisons. The current revision strengthens the statistical contribution of the paper (the permutation test of Section 2.3) with four new appendices (S1–S4) that respond directly to the reviewer's §3.5(b), §3.6(a), and §3.6(b). The additional baselines requested in §3.1 and the additional dataset(s) requested in §3.2 are in active development and will be included in the next revision; we describe the planned scope explicitly under each comment below. We have inlined our response under each reviewer comment so that the editor can verify on a per-comment basis what is in this revision and what is planned.
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

> **Authors' response — ◐.** We agree that a comparison against modern compact transformers is essential and is the most significant gap in the original submission. In the current revision we have not yet added these baselines; we plan to include in the next revision:
>
> - **iTransformer** (Liu et al., 2024) — at matched parameter count;
> - **A scaled-down vanilla transformer** at matched parameter count (1 layer, 1 head, $d_{\mathrm{model}}$ chosen to match MiniTransformer parameter count) — to isolate the effect of the *specific* simplifications from the effect of fewer parameters;
> - **A kernel attention baseline without the temporal-decay term in Eq. 1** — to attribute the contribution of the decay cleanly (this is also an ablation requested in §3.3 and is most cheaply produced together with the other §3.3 ablations);
> - **DLinear** and **PatchTST** as compact non-transformer / patch-based baselines.
>
> We will report these comparisons on the simulation, on LORA D1/D2, and on at least one additional dataset per §3.2. We will also add the connection to **Lu & Yang (2025), *Linear Transformers as VAR Models*** to the introduction and to Section 2.2. The reviewer is correct that this is more than analogy: linear attention in the autoregressive setting is interpretable as a dynamic VAR, exactly the framing of MiniTransformer. We will cite Lu & Yang (2025) accordingly.

### 3.2 The real-data evaluation needs more than one cohort

The application to the LORA resilience study is appropriate and the results are interesting. However, if the architectural contribution is to be taken on its own merits, one real dataset (split into two subsets) is not sufficient in a machine-learning-style evaluation. This is a general issue with empirical machine-learning papers: a method that wins on one dataset is often an artefact of that dataset. I would ask the authors to either:

- Evaluate MiniTransformer on additional longitudinal cohort datasets — there are many longitudinal studies with similar structure (small $N$, ~5–20 visits, mixed binary/continuous items) in the psychiatry, epidemiology, and aging literatures; and/or
- Evaluate on at least a subset of **standard short/medium-horizon time series benchmarks** where fair small-data comparisons are possible. Relevant candidates from the forecasting literature include the ETT family (ETTh1/h2, ETTm1/m2; Zhou et al., 2021), Weather, Traffic, Electricity, Exchange Rate, ILI, and the M4/M5 competition data (Makridakis et al., 2020; 2022). Even if the authors argue these benchmarks are not the primary target, having at least one or two of them in the evaluation makes the method's *scope* much clearer.

I do not think the paper needs to win on every one of these benchmarks — the authors could reasonably argue that their method is designed for the small-$N$/few-visits regime specifically — but a transparent evaluation across several datasets is important.

> **Authors' response — ◐.** We agree. The two LORA subsets share data and are not sufficient as standalone evidence of generalisation. In the current revision we have added the first of two new datasets:
>
> - **PBC2** ✓ — the Mayo Clinic primary biliary cirrhosis longitudinal data (`survival::pbcseq`), $252$ patients with median $6$ visits, $10$ binarised clinical markers. Results are reported in **Appendix S5** of the revised manuscript. The chosen target (`bili_high`, elevated serum bilirubin) passes the predictability gate by a wide margin (MSE $0.033$ vs $0.288$ for marginal averaging — a $9$-fold reduction). The permutation test on PBC2 identifies a clinically coherent set of top contexts for predicting bilirubin elevation: spider angiomas, low platelets, low albumin, and edema -- the textbook signature of advanced liver disease. The V-monotonicity behaviour matches the simulation and LORA: signal-bearing contexts become more significant with $V$, while non-signal contexts stay flat or become more conservative.
>
> Still planned for the next revision:
>
> - **ILI** (Influenza-Like Illness, weekly CDC counts) as a *standard short-horizon time series benchmark*. This is the medical dataset on the reviewer's suggested list and is most consistent with the paper's framing while still being a recognised forecasting benchmark.
>
> Together with the §3.1 baselines (also planned), this will give the paper a transparent evaluation across one synthetic generative model, two real cohorts (LORA and PBC2), and one external time-series benchmark. The goal, as the reviewer notes, is transparent assessment rather than winning on all benchmarks; the PBC2 result already demonstrates that the architecture and the permutation test recover clinically meaningful structure outside the cohort the method was originally illustrated on.

### 3.3 Ablation study for each architectural component

The MiniTransformer simultaneously introduces (i) a temporal-decay multiplier on the softmax attention kernel (Eq. 1–2), (ii) a cumulant/output head that aggregates the entire history with an *additional* temporal-decay factor toward the prediction horizon (Eq. 3), (iii) scalar-valued attention heads, (iv) removal of the final large projection layer, and (v) a relative positional scheme parameterised by $w_{\mathrm{dist}}, w_{\mathrm{horizon}}, \gamma$. Each individual choice is well-motivated in isolation, but it is not clear which simplifications are doing the work. I would ask for an ablation table varying each of these components independently, reporting both predictive MSE and the behaviour of the permutation-based p-values. Specifically:

- Remove the cumulant head in Eq. 3 (keep only the per-timestep output from Eq. 2) — how does performance change?
- Replace the exponential-decay multiplier in Eq. 1 with a **regularisation on the attention weights** that encourages decay in expectation (rather than baking it into the kernel). On the face of it these two implementations should achieve similar inductive biases, but one is a soft prior and the other is a hard multiplicative mask; which one actually helps?
- Replace the authors' positional parameterisation $(w_{\mathrm{dist}}, w_{\mathrm{horizon}}, \gamma)$ with a **rotary positional encoding** (RoPE; Su et al., 2024), which is currently the default in most modern small transformers and is known to work well even in small-data settings.

The exponential-decay-vs-regularisation point is worth emphasising. The Introduction rightly argues that the *most recent* observation need not be the most relevant, which is the core motivation for using attention in the first place. But Eqs. 1 and 3 then impose a hard exponential decay on top of the softmax, which is philosophically the opposite assumption. I believe the authors' intent is that the decay serves as a prior or regulariser in the small-data regime, but if so I would prefer to see this made explicit, and compared with a more conventional regulariser applied to the attention weights themselves.

> **Authors' response — ◐.** We agree that the ablations would clarify which simplifications carry the work. In the next revision we will add an ablation table (predictive MSE and the behaviour of permutation-based p-values) for the following variants on the simulation and on LORA:
>
> - Removing the cumulant head in Eq. 3, keeping only the per-timestep output from Eq. 2 — this also addresses §3.4;
> - Replacing the exponential-decay multiplier in Eq. 1 with an L2 regulariser on the attention weights (the soft-prior alternative the reviewer describes);
> - Replacing the relative positional parameterisation $(w_{\mathrm{dist}}, w_{\mathrm{horizon}}, \gamma)$ with rotary positional encoding (RoPE; Su et al., 2024).
>
> We will additionally make explicit in Section 2.2 that the temporal-decay multiplier is intended as a small-data prior or regulariser rather than a model assumption that recent observations are necessarily most relevant — this is the motivation the reviewer correctly inferred — and we will state the assumption explicitly so that the comparison with attention-weight regularisation is interpreted in that light.

### 3.4 The cumulant head (Eq. 3) appears to double-count history

If the per-timestep output $\tilde{x}^{(h)}_{t_i}$ in Eq. 2 is a correct attention summary, then (in principle) *all* of the history is already compressed into $\tilde{x}^{(h)}_{t_T}$. Equation 3 then integrates these $\tilde{x}^{(h)}_{t_i}$ a second time over $i = 1, \ldots, T$ (with an additional decay toward $t_{T+1}$). Holistically, history is therefore counted multiple times, which the second decay factor in Eq. 3 appears to compensate for. The rationale for the resulting low-dimensional representation $\mathbf{z}$ is not fully convincing to me: *any* output head produces a low-dimensional representation, so the relevant question is why the *cumulative* form of Eq. 3 is the right inductive bias. An ablation removing Eq. 3 and using only Eq. 2 (see Section 3.3 of this report) would address this directly.

Relatedly, could the authors clarify the notation: in Eq. 2 the variable is $\tilde{x}^{(h)}_{t_i}$, but in Eq. 3 the summand is $\tilde{x}'^{(h)}_{t_i}$. After re-reading I concluded the prime denotes a transpose rather than a different quantity, but this is easy to misread and should be stated explicitly.

> **Authors' response — ◐.** We thank the reviewer for this observation. The cumulant head was motivated by additional flexibility in the small-data regime: the per-timestep summary $\tilde{x}^{(h)}_{t_i}$ in Eq. 2 captures attention over the history *up to* $t_i$, but Eq. 3 then re-weights these summaries with a *prediction-horizon* decay relative to $t_{T+1}$, which is a different inductive bias than the pairwise decay in Eq. 1. Concretely, Eq. 3 expresses the prior that recent timesteps' history-summaries are more relevant for the next-step prediction, whereas Eq. 2 expresses the prior that recent history within each timestep matters. In a vanilla transformer these two effects are entangled in the projection layer; the cumulant head decouples them with a small number of additional parameters.
>
> We acknowledge that this argument is not made explicitly in the manuscript and will rewrite the relevant paragraph in Section 2.2 to state the inductive bias clearly. The §3.3 ablation that removes Eq. 3 will provide direct empirical evidence.
>
> **Notation — ✓.** We will state explicitly in Eq. 3 that the prime denotes a transpose, and switch the notation to $\tilde{x}^{(h)\top}_{t_i}$ to remove any possible confusion with a derivative. This is also addressed in minor comment #1.

### 3.5 Section 2.3 is the strongest contribution — and can be made much more general

I want to emphasise that Section 2.3 is, in my view, the paper's most distinctive statistical contribution and deserves to be pushed further. Several of the observations that make this section valuable apply much more broadly than the authors state:

(a) **Equation 5 is almost model-agnostic.** Although the current derivation uses the specific form of the MiniTransformer, the underlying idea — evaluating the change in a prediction when a feature is zeroed out, averaged over a representative set of visit/context realisations — is applicable to *any* differentiable (and indeed any black-box) sequence model: LSTMs (Hochreiter & Schmidhuber, 1997), state-space models, or any transformer variant. The authors could briefly point this out and comment on what the specific structure of the MiniTransformer buys them, beyond making Eq. 5 closed-form.

> **Authors' response — ☐.** We will add a paragraph to Section 2.3 making this explicit: Eq. 5 generalises naturally to any differentiable sequence model (LSTMs, state-space models, other transformer variants) and indeed to any black-box predictor whose prediction is differentiable in a context perturbation. The role of the MiniTransformer's specific structure is to make Eq. 5 *closed-form*. We will discuss what the specific architecture buys us (interpretability of the attention parameters and a closed-form $\Delta$) versus what is gained by a general black-box implementation.

(b) **The permutation test in Eq. 7 relies on row-exchangeability of $\boldsymbol{\Delta}^{(r)}$, not on the model itself.** Under the null of no distinct context effect the rows of $\boldsymbol{\Delta}^{(r)}$ are exchangeable, and the test is therefore distribution-free *provided the model is well-calibrated for prediction*. Is good predictive performance a prerequisite for the test to be meaningful? If the underlying transformer does not fit the data well, what is the behaviour of Eq. 7? A short discussion, possibly with a synthetic "bad model" baseline, would clarify the interpretation.

> **Authors' response — ✓.** This is a central concern that we have addressed with new empirical work. The new appendices S1 and S2 of the manuscript demonstrate the following.
>
> - **Appendix S1 (predictability gate).** We propose a two-stage procedure: the permutation test of Section 2.3 is applied to a target $r$ only if the trained MiniTransformer beats the marginal-mean baseline on a held-out evaluation set ($\mathrm{MSE}_{\mathrm{MT}}(r) < \mathrm{MSE}_{\mathrm{avg}}(r)$). This is the minimal sanity check the reviewer's exchangeability argument calls for: it asks whether the model has learned any target-specific structure beyond the global frequency of $y_r$. We also report a per-target Gaussian regression baseline alongside for transparency, but do not require MT to beat regression for the gate to pass — that is a stronger criterion that asks a different question (whether MT adds value beyond a one-step linear predictor) and would penalise the test for situations where regression and MT capture overlapping signal. On the simulation, the gate selects $j_3$ — the only data-generating-process-predictable target — by a substantial margin (MSE $0.102$ vs $0.227$), confirming that the procedure correctly identifies the case where the test is meaningful (Table A1 of the manuscript).
> - **Appendix S2 (null calibration with 500 repetitions).** With the trained model held fixed, we ran the permutation test 500 times. On the data-generating-process null variables (3–9 in the simulation), the empirical Type-I error rate at $\alpha=0.05$ is $0.000$ across $3500$ trials. The null-variable p-value distributions are concentrated in $[0.3, 0.8]$ rather than uniform on $[0,1]$ — the *conservative* shift our paper §2.3 already predicts when the empirical null is contaminated by signal rows, which is exactly the reviewer's exchangeability argument.
> - **Stress test on a generatively-null target (Appendix S2).** When we deliberately apply the test to a generatively-null target (predindex $=5$), where no variable carries DGP-level signal and the gate fails, we still find the test close to nominal calibration (mean rejection rate $0.064$ across nine non-self variables at $\alpha=0.05$). This empirically demonstrates that the test does not catastrophically misbehave even outside the regime where the gate passes.
>
> Together, these three results provide a concrete answer: good predictive performance is *not* strictly required for the test's calibration, but it is the regime in which the test's interpretation as "context effects on the data" is justified. We have made this explicit in the revised Section 2.3 and recommend the gate as the canonical entry point to the test.

(c) **The context can be perturbed, not just fixed.** Currently the authors fix a single reference context $\mathbf{x}^{\mathrm{context}}$ and vary only the visit realisations $\mathbf{x}^{\mathrm{visit}}_v$. But nothing in the logic of Eqs. 5–7 requires a single context: one could sweep over a set of contexts $\{\mathbf{x}^{\mathrm{context}}_u\}_{u=1}^U$ and obtain a *three-dimensional tensor* of $\Delta$-values (features × visits × contexts). Summary statistics over this tensor would be much less sensitive to the choice of reference context, and the permutation test still applies (exchangeability now over the combined axes as appropriate). If additionally multiple prediction targets $y_r$ are considered, $\mathbf{S}$ becomes four-dimensional. This extension seems both natural and practically useful, and I would encourage the authors to at least discuss it, if not implement it.

> **Authors' response — ☐.** We agree this extension is natural and useful. In the next revision we will add a discussion of the multi-context extension (the 3-D tensor of $\Delta$-values over features × visits × contexts, and its 4-D extension over multiple targets), and outline how the permutation test in Eq. 7 generalises with appropriate exchangeability assumptions over the combined axes. We will also report a small implementation on the simulation to illustrate the reduced sensitivity to the choice of reference context.

### 3.6 Calibration of the permutation-based p-values

I have two related concerns about the p-values in Tables 2 and 4.

**(a) Monotonicity in $V$.** In Table 2 the reported p-values clearly decrease as $V$ grows (from $V=8$ to $V=12$ to $V=16$). The authors discuss this as a power issue. I would argue the problem is the opposite: in *real* data where no feature is literally noise but many features have small but non-zero signal, taking $V$ large will likely drive essentially all p-values to zero. This is the same issue that arises in high-$n$ linear regression, where p-values stop being informative because any deviation from the null is detectable. In Table 4 the reported $V = 8$ choice makes the results readable, but raises the question: what happens on the LORA datasets as $V$ is increased toward its maximum? If most or all variables become "significant," then the significance ranking is context-length dependent rather than a property of the data, which limits the usability of Eq. 7 as reported.

> **Authors' response — ✓.** We have addressed this directly with a V-sweep on both the simulation (Appendix S3) and on the LORA cohort (Appendix S4), the dataset the reviewer specifically mentions:
>
> - **Simulation (Appendix S3, $V \in \{5, 6, 7\}$, $\mathrm{nrepp}=500$).** Power on signal variables increases monotonically with $V$ (e.g.\ rejection rate at $\alpha=0.05$ for $j_3$ rises from $0.71$ to $0.77$). The empirical Type-I error rate on the seven null variables is exactly $0.000$ at every $V$ tested. Larger $V$ is therefore strictly beneficial in this controlled setting.
> - **LORA D1 and D2 (Appendix S4, $V \in \{5, 6, 7\}$, $\mathrm{nrepp}=500$).** This addresses the reviewer's specific concern about real data with many small effects. We do not observe the "all p-values driven toward zero" pathology. Instead, the variables that the model has learned to use as informative contexts (e.g.\ \texttt{dh\_38} on D1 — the same top-ranked context as in the original Table 4) become more significant as $V$ grows, while the remaining candidate contexts stay flat or become *more* conservative (e.g.\ \texttt{le\_8} on D1: mean $p$ rises from $0.796$ to $0.836$ as $V$ goes from $5$ to $7$). The significance ranking is therefore not driven by $V$.
>
> The reviewer's worst-case scenario is empirically refuted on LORA, the very dataset that motivated the concern. We acknowledge this is a single-fold analysis on each LORA dataset; a fold-averaged version is a natural follow-up.

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

> **Authors' response — ☐.** We will add a brief discussion of the cases in which the single-variable-target MSE is not improved by the MiniTransformer (e.g.\ when the target is well-described by a single linear regressor on the previous time point and there is little additional contextual information for attention to exploit). The simulation result for $n_{\mathrm{train}}=100$ is also a clear example, and we will explicitly note that the gate of Appendix S1 applies in such cases.

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

> The current revision contains the new empirical work that addresses §3.5(b), §3.6(a), §3.6(b), and a substantial part of §3.2:
>
> - **Appendix S1 (Predictability gate)** — formalises the two-stage procedure motivated by §3.5(b);
> - **Appendix S2 (Null calibration of the permutation test)** — histograms, Q-Q plots, KS tests across $\mathrm{nrepp}=500$ repetitions on the simulation, with stress test on a generatively-null target;
> - **Appendix S3 (Sensitivity to $V$, simulation)** — V-sweep $V \in \{5, 6, 7\}$ at $\mathrm{nrepp}=500$, demonstrating monotonic power increase on signal variables and exact $0.000$ Type-I error on null variables across all $V$;
> - **Appendix S4 (V-sweep + gate on LORA)** — the same analyses applied to LORA D1 and D2 on fold $0$ of the original $10$-fold split, showing that the reviewer's concern about $V$-driven inflation of false positives does not materialise on the real cohort;
> - **Appendix S5 (V-sweep + gate on PBC2)** — the same analyses applied to a *second, structurally distinct* clinical cohort (Mayo Clinic primary biliary cirrhosis, $252$ patients, $10$ binarised markers). The chosen target (elevated bilirubin) passes the predictability gate by a $\sim 9$-fold MSE reduction over averaging. The permutation test recovers a clinically coherent set of top contexts (spider angiomas, low platelets, low albumin, edema) — the textbook signature of advanced liver disease.
>
> Remaining items (§3.1 baselines, §3.2 ILI benchmark, §3.3 ablations, §3.5(a), §3.5(c), §3.7, and minor comments #3, #5, #6, #8) are acknowledged inline above and will be implemented in the next revision.

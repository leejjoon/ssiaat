# Literature-sourced PN emission-line list for the SPHEREx 0.75–5.0 μm band

Research note supporting `examples/fit_pn_lines.py`'s `LINE_LIST` and
`ssiaat.model.sed.get_pah()`. Written to the same standalone-analysis
convention as `ssiaat_improve_plan.md` / `table_size_possible_reduction.md`
at the repo root, but scoped to `examples/`.

**Purpose.** The current `LINE_LIST` in `fit_pn_lines.py` (H I Paschen/
Brackett/Pfund lines, He I 1.08303/2.05869, [S III] 0.90686/0.95306) and the
PAH shapes in `sed.py::get_pah()` (aromatic 3.2481–3.3382 μm, aliphatic
3.32–3.52 μm) came from the researcher's own memory, not a citation. This
note traces every one of those numbers back to a real source (or states
plainly that it couldn't be), and compiles additional literature-supported
PNe lines in 0.75–5.0 μm as fitter candidates.

**How the search was done.** WebSearch/WebFetch against NIST ASD, ADS,
arXiv, and A&A/MNRAS/ApJ full text; PDFs that WebFetch couldn't decode
(binary/compressed) were re-read directly with the `Read` tool, which
handles PDF text extraction natively. H I vacuum wavelengths were also
independently computed from the Rydberg–Ritz formula with the CODATA
reduced-mass hydrogen Rydberg constant, then cross-checked digit-by-digit
against three independent astronomical calibration compilations (see
Table 1) and against a 2023 JWST paper's tabulated line-list wavelengths —
this cross-check is what caught (and let me correct) at least two digit-
transposition errors introduced by WebFetch's own summarization of NIST's
raw HTML tables (e.g. a NOT/IAC page rendering the Brackett-alpha
wavelength as "4.5225" when the same page's quoted frequency column,
2467.765 cm⁻¹, algebraically requires "4.0522*"). Wherever a number in this
note rests on a single AI-summarized web page rather than a value I could
verify by an independent method (formula, second source, or internal
frequency/wavelength consistency), that is stated explicitly.

---

## 1. Air vs. vacuum convention

NIST ASD's own stated policy (https://physics.nist.gov/PhysRefData/ASD/lines_form.html,
confirmed by direct query in this session) is:

> "λ in: vacuum below 200 nm, air between 200 and 2000 nm, vacuum above 2000 nm"

i.e. **air is used only for 0.2–2.0 μm; vacuum below 0.2 μm and above 2.0 μm.**
This is the opposite emphasis from how the task brief summarized it, but is
NIST's literal, primary-source statement, and I take it as authoritative
for what "the standard convention" means.

In practice, near-IR *astronomical* calibration references (UKIRT's and the
Nordic Optical Telescope's published H-recombination-line tables, see
Table 1) do **not** apply NIST's 2 μm split — they tabulate vacuum
wavelengths uniformly across J/H/K/L, citing intensities (not wavelengths)
from Hummer & Storey (1987). The JWST MIRI λ_lab column in Jones et al.
(2023, see §5) is also vacuum, consistent with the Rydberg-formula value,
for every H I line in their table (all ≥4.9 μm, so this doesn't test the
<2 μm regime, but it's the most recent primary confirmation available that
"vacuum, Rydberg-formula value" is the value real PN spectroscopy papers
actually tabulate as λ_lab).

**Net conclusion used throughout this note:** for H I recombination lines,
I report the vacuum wavelength (from the Rydberg formula, self-consistent
with every external check I could run) as the primary number, and note the
NIST-air equivalent for lines <2 μm as a secondary column, because both
conventions are in real, current use and the repo's own numbers turn out to
be an inconsistent mix of the two (§7).

---

## 2. H I recombination lines (Paschen, Brackett, Pfund, Humphreys) in 0.75–5.0 μm

Computed from the Rydberg–Ritz formula
σ = R_H·(1/n₁² − 1/n₂²), λ_vac = 1/σ, using the CODATA reduced-mass hydrogen
Rydberg constant R_H = R_∞/(1 + mₑ/M_p) = 109677.583 cm⁻¹ (R_∞ = 10,973,731.568160 m⁻¹,
mₑ/M_p = 1/1836.15267343, CODATA 2018). This is exactly the same physics
NIST ASD's own "Ritz wavelength" column uses for H I in this regime (no
resolvable QED/fine-structure correction at 5-significant-figure precision
for n≥4→n≥3 transitions). Air values below use the Peck & Reeder (1972)
standard-air dispersion formula, the same one documented as NIST ASD's air-
conversion method.

Cross-checks: NOT/IAC NOTCAM calibration pages (Paschen, Brackett, Pfund
series, http://www.not.iac.es/instruments/notcam/ReferenceInfo/, citing
Hummer & Storey 1987, MNRAS 224, 801 for intensities) match every value
below to 4–5 significant figures once I corrected the digit-transposition
extraction errors described above by cross-checking each page's quoted
frequency column. Jones et al. (2023, MNRAS 523, 2519) Table 1 independently
confirms the same vacuum values for the members that fall in their ≥4.9 μm
MIRI range (Pfund-α = 7.460 μm, Humphreys-α = 12.372 μm — both computed here
to 7.45988/12.37193, matching to the quoted precision).

| Line | λ_vac (μm) | λ_air (μm), if <2 μm | Confidence |
|---|---|---|---|
| Pa-δ (7→3) | 1.00522 | 1.00492 | confirmed: Rydberg calc + NOT/IAC table |
| Pa-γ (6→3) | 1.09412 | 1.09382 | confirmed: Rydberg calc + NOT/IAC table |
| Pa-β (5→3) | 1.28217 | 1.28182 | confirmed: Rydberg calc + NOT/IAC table |
| Pa-α (4→3) | 1.87563 | 1.87512 | confirmed: Rydberg calc + NOT/IAC table + UKIRT-family compilations |
| Br-η (10→4) | 1.73669 | 1.73621* | Rydberg calc only, not cross-checked against a table for this member |
| Br-ε (9→4) | 1.81792 | 1.81742* | Rydberg calc + NOT/IAC table |
| Br-δ (8→4) | 1.94510 | 1.94456 | confirmed: Rydberg calc + NOT/IAC table + Hora et al. (1999, see §5) |
| Br-γ (7→4) | 2.16613 | — (>2 μm, vacuum per NIST) | confirmed: Rydberg calc + NOT/IAC table + Hora et al. (1999) |
| Br-β (6→4) | 2.62588 | — | confirmed: Rydberg calc + NOT/IAC table |
| Br-α (5→4) | 4.05228 | — | confirmed: Rydberg calc + NOT/IAC table (frequency-column cross-check) |
| Pf-γ (8→5) | 3.74057 | — | confirmed: Rydberg calc + NOT/IAC table |
| Pf-β (7→5) | 4.65379 | — | confirmed: Rydberg calc + NOT/IAC table |
| Pf-α (6→5) | 7.45988 | — | out of range (>5 μm); confirmed against Jones et al. (2023) Table 1 |
| Hu-θ (14→6) | 4.02088 | — | Rydberg calc only; Humphreys members this faint/blended are not
individually reported in any PN paper I found — theoretical only |
| Hu-ε…Hu-π (11→6…22→6) | 4.67…3.55 | — | Rydberg calc only, theoretical; not confirmed as individually resolved
in any real PN spectrum I found |

\* Peck–Reeder air formula extrapolated beyond its usual calibration range;
flagged as lower-confidence than the vacuum column for that reason.

Additional Paschen/Pfund members exist between the ones tabulated and each
series limit (Paschen limit 0.8206 μm, Pfund limit 2.2794 μm, Humphreys
limit 3.2824 μm — all by the same Rydberg formula) but blend together and
are not individually reported as resolved features in the PN spectroscopy
papers surveyed here; I list the named α–ε/ζ/η members that a real fitter
would plausibly resolve and leave the rest as a formula-only continuum.
Humphreys-series members below Hu-δ (10→6, 5.129 μm, itself just outside
5.0 μm) fall in range but I found **no PN paper reporting any of them
individually** — they are Rydberg-formula theoretical values only, not
confirmed detections, and are not recommended as fitter candidates without
independent confirmation.

---

## 3. He I / He II lines actually reported in real PNe near-IR spectra

| Line | λ (μm) | Convention | Source | Commonality |
|---|---|---|---|---|
| He I 1.0830 (2s³S–2p³P, blend of 3 components spanning ~1.2 Å) | 1.08303 (air, blended) | air (blend of 10829.09/10830.25/10830.34 Å air components) | Multiple stellar/solar-physics compilations (e.g. Kramida et al./NIST ASD He I levels, cross-checked via WebSearch, not independently re-verified against raw NIST ASD output in this session) | Extremely common, one of the strongest non-H lines in PNe; explicitly used as a PN diagnostic line in Rudy et al. (2001, IC 5117) and Hora et al. (1999) |
| He I 1.7002 | 1.7002 | not stated by source | Hora, Latter & Deutsch (1999, ApJS 124, 195 — arXiv:astro-ph/9904202): explicitly named as present "in some PNe" in the H band alongside [Fe II] 1.6440 | single-source in this search; H-band, less universal than 2.058/2.112 |
| He I 2.0587 (2p¹P°–2s¹S) | 2.05869 | ambiguous (right at the NIST 2 μm crossover; multiple compilations just say "2.0587") | UKIRT calibration pages (Helium Lines, 1.80–2.05 μm range) + Hora et al. (1999) explicitly: "the brightest line in the Ks band is Brγ, with He I 2.058, 2.112 μm" | confirmed against 2 independent sources; one of the most-cited He I PNe lines |
| He I 2.1126 / 2.1138 (4S–3P) | ~2.1126–2.1138 | not stated by source | UKIRT compilation (per WebSearch summary only — not independently re-verified against the raw page); corroborated qualitatively (not numerically) by Hora et al. (1999)'s "2.112" | single clean-source confirmation; note this sits ~0.009 μm from H₂ 1–0 S(1) at 2.1218 — a real near-blend risk for any fitter that includes both |
| He II lines (generic) | not resolved to a specific line here | — | Rudy et al. (2001, IC 5117): abstract states He II is present among "a wide range of ionization" lines detected, but I did not obtain the paper's line-list table (ResearchGate blocked, 403) to pin an exact wavelength | **unable to verify a specific He II wavelength for PNe in 0.75–5 μm from a source I could actually read**; SOFIA/FLITECAM paper (Smith & McLean 2007, arXiv:0712.0812, read in full) explicitly labels an He II line at ≈3.09 μm in NGC 7027's spectrum (Fig. 1) — this is the one He II wavelength in this range I *can* confirm from a source I fully read |

---

## 4. Forbidden lines commonly detected in real PNe near/mid-IR spectra

| Line | λ (μm) | Convention | Source(s) | Commonality / confidence |
|---|---|---|---|---|
| [S III] 9069 | 0.90686 (≈9068.6–9069 Å across sources) | air (0.2–2 μm NIST regime) | Multiple deep PN abundance papers found via search cite [S III] λλ9069,9531 as the standard pair (e.g. review discussion citing Storey & Zeippen 2000 for the fixed 2.48 theoretical intensity ratio I(9531)/I(9069)); I was not able to fully read a primary table with digit-level precision (arXiv PDFs for NGC 5315-type papers returned unreadable to WebFetch) | confirmed against 2+ independent search results converging on 9068.6–9069 Å; extremely common, bright, and telluric-absorption-free in PNe — one of the most standard PN diagnostic pairs in the optical/near-IR literature |
| [S III] 9531 | 0.95306 (≈9530.6–9532 Å across sources) | air | same as above | confirmed to ~1 Å precision across sources (9531 vs 9532 Å spread is within normal source-to-source rounding); commonality as above |
| [C I] 0.8727 (5→4) | 0.8727 | not stated (optical/near-IR convention, presumably air) | Danziger & Goad (1973, first detection in NGC 7027); Liu et al. (1995); **Akras et al. (2024, A&A, arXiv:2409.00371)** — read in full; their Table 1 lists ~40 PNe with confirmed [C I] λ8727 and/or λλ9824,9850 detections, including IC 5117 | confirmed against 3 independent sources; genuinely common — not currently in the repo's list, a strong candidate addition |
| [C I] 0.9824, 0.9850 (4→3 doublet) | 0.9824, 0.9850 | as above | same as above | confirmed, same source set; sits close to the [S III] 9531 feature — a real near-blend consideration for any fitter covering band 1 |
| [Fe II] 1.257 | 1.257 | not stated | Hora et al. (1999); Akras et al. (2024, A&A 689, A70 — the dedicated "[Fe II] 1.644 μm imaging survey of PNe with low-ionisation structures" paper, found via search, abstract/summary only, not fully read) | confirmed against 2 independent sources; well-established shock/LIS (low-ionization-structure) tracer, not a "core" nebular line — present in a subset of PNe with shocks/knots, not universal |
| [Fe II] 1.644 | 1.644 | not stated | same as above | same confidence; the brighter/more commonly imaged of the [Fe II] pair, used as R(Fe)=[Fe II]1.644/Brγ shock diagnostic |
| [Si VI] (coronal) | 1.964 (also seen quoted 1.9630/1.9641) | not stated | Casassus, Roche & Barlow (2000, MNRAS 314, 657, coronal-line survey of NGC 6302/6537, 3–13 μm) — abstract/summary read, not the full table; corroborated by Dinerstein review (NED, read in full) which independently lists "[Si VI] 1.96 μm" | confirmed against 2 sources; **rare** — coronal lines require an unusually hot (≳140,000–180,000 K) central star, so only appear in a minority of PNe (NGC 6302, NGC 6537, and similar "coronal-phase" objects), not expected as a generic core-list line |
| [Si VII] (coronal) | 2.48 | not stated | Dinerstein review (read in full) | single-source in what I could read; same rarity caveat |
| [Si IX] (coronal) | 3.934 (also 3.92 in the Dinerstein review's rounded citation) | not stated | Casassus, Roche & Barlow (2000) abstract explicitly: "[Si IX] 3.934 μm"; Dinerstein review independently: "[Si IX] 3.92 μm" | confirmed against 2 sources to 3 significant figures; same rarity caveat — described by Dinerstein as "record-setting" ionization for a PN coronal line |
| [Ni II] | 0.7378 | not stated | Search results citing detections in NGC 3242, NGC 7009, NGC 6153, NGC 6369, Tc 1 (multiple papers, not individually read in full) | **out of the requested 0.75–5.0 μm range** (0.7378 μm is just below the 0.75 μm floor) — noted for completeness, not a fitter candidate here |
| [N I] | ~0.8629 / 0.8729 (as reported by one search summary) | not stated | Single WebSearch summary only; I could not independently verify this against a source I read directly, and the 0.8729 value is suspiciously close to (and may be a confusion with) [C I] 0.8727 above | **unable to verify** — flagged explicitly rather than asserted; do not add to a fitter on this evidence alone |

---

## 5. Molecular / dust features

### PAH 3.3 μm complex

The best PN-specific source I could read in full is **Smith & McLean (2007,
ApJ, arXiv:0712.0812)**, "A Survey of 3.3 μm PAH Emission in Planetary
Nebulae" (SOFIA/FLITECAM commissioning survey, R≈1700, 20 objects including
IC 5117, NGC 7027, BD+30°3639 — several of the same objects as the Rudy
near-IR papers):

- Literature reference shape (Tokunaga et al. 1991; van Diedenhoven et al.
  2004), which Smith & McLean's own PNe sample matches: **Class A3.3**
  emission — Gaussian-like profile centered at **3.290 μm**, FWHM
  **0.040 μm**. All 11 PAH-detected PNe in their sample were classified
  Class A3.3 (vs. Class B3.3, centered 3.293 or 3.297 μm with FWHM
  0.037 μm, typical of H II regions/reflection nebulae, *not* PNe).
- Smith & McLean's own measured sample average: center **3.288 ± 0.002 μm**,
  FWHM **0.042 ± 0.002 μm**, across M3−35, NGC 7027, HB 5, IC 5117,
  BD+30°3639, M2−43, IRAS 21282+5050, Hen 2−459, NGC 6886, NGC 6790, M4−18.
  IC 5117 individually: center 3.289 ± 0.002 μm, FWHM 0.038 ± 0.005 μm.
- They also explicitly identify a **real H I line sitting on top of the PAH
  feature**: Pfund-δ at **3.296 μm** (their measured value — matches the
  3.29700 μm Rydberg-formula value in §2 to 3 decimal places) — seen in
  emission in HB 5, M3−35, BD+30°3639, and NGC 7027, and subtracted out
  before fitting the PAH band itself in their analysis. Pfund-ε (3.039 μm)
  and He II (≈3.09 μm) are also identified in NGC 7027's spectrum
  specifically (their Fig. 1).
- Roche et al. (1996, MNRAS 280, 924) and Cohen & Barlow (2005, MNRAS 362,
  1199) established a C/O ≳ 0.6 threshold for PAH emission in PNe — Smith
  & McLean's own fit gives 0.65 ± 0.28, consistent.

**Aliphatic 3.4–3.6 μm component**: I found repeated general statements
that this feature exists and is described in the literature as spanning
roughly **3.4–3.6 μm**, and that it tends to be *weaker in PNe than in
proto-PNe* (an evolutionary aliphatic→aromatic transition) — but I did **not
find a PN-specific paper giving a precise central wavelength/FWHM the way
Smith & McLean did for the 3.3 μm aromatic band.** This is confidence level
"general literature statement, not confirmed against a specific PN
measurement" — see §7 for how this compares to the repo's 3.32–3.52 μm.

### H₂ rotational-vibrational lines

| Line | λ (μm) | Source | Confidence |
|---|---|---|---|
| H₂ 1–0 S(1) | 2.1218 | Ubiquitously cited across every near-IR PN paper read in this session (Hora et al. 1999; Rudy et al. 2001 abstract; Kastner-type papers found via search) as *the* brightest, most easily detected H₂ line in PNe | confirmed against 3+ independent sources; **not currently in the repo's `LINE_LIST`** — strong candidate addition, with the caveat (§3) that He I ≈2.1126–2.1138 sits only ~0.008–0.009 μm away |
| H₂ 2–1 S(1) | 2.2477 | Cited (ratio-diagnostic pairing with 1–0 S(1)) in a search-result summary of an H₂-in-LIS PNe paper | single-source in what I read; standard H₂ excitation-diagnostic line, real but not independently re-verified here |
| H₂ 0–0 S(2)…S(7) | mid-IR, mostly >5 μm | Mata et al. (2016, MNRAS 459, 841, Spitzer IRS survey of 157 compact Galactic PNe — abstract/summary read) | These fall in the Spitzer IRS 5.2–14.5 μm band, i.e. **outside** the 0.75–5.0 μm SPHEREx range; mentioned only as context that H₂ mid-IR lines are a real, common PN feature just beyond this band |
| H₂ v=1→0 Q(1), Q(3) | ~2.4 μm region | Hora et al. (1999), read in full, mentions tentative Q-branch H₂ detections (e.g. IC 2003) at low S/N | low confidence per the source itself ("detected at roughly 2σ level... not reliable") |
| CO first-overtone bandhead | ~2.3–2.5 μm | Hora et al. (1999), read in full: explicitly reports "CO bandhead emission at λ = 2.3 – 2.5 μm" in M1-92 (a proto-PN) at low S/N | single source; noted as present in H₂-dominated/proto-PN objects, not a general core-PN feature |

### Ices (H₂O, CO₂, CO)

- **H₂O ice**: Cohen, Barlow, Sylvester et al. (1999, arXiv:astro-ph/9901126,
  read in full) report water ice bands in the carbon-rich PN CPD−56°8032
  at **43 and 62 μm** — far-infrared, **not** in the 3 μm O–H-stretch region
  and **not** in the 0.75–5.0 μm SPHEREx range. A WebSearch summary
  (not independently verified) adds NGC 6537 and NGC 6302 as the only other
  PNe with confirmed H₂O ice, with the same caveat about wavelength.
- **CO₂ ice**: A very recent paper, "Detection of CO₂ ice in the planetary
  nebula NGC 6302" (2026, A&A, arXiv:2602.22366 — found via search, not
  independently read in full) reports the detection via the ~15 μm
  *bending*-mode double-peaked absorption profile (14.9–15.15 and
  15.2–15.3 μm), **not** the 4.27 μm stretching mode; a search summary
  separately notes that CO₂'s 4.27 μm stretching mode was historically
  first detected by ISO in other object classes, but I found **no PN-
  specific report of the 4.27 μm CO₂ band** — so within the SPHEREx
  0.75–5.0 μm range, I cannot confirm a CO₂ ice detection in any PN.
- **CO fundamental band (~4.3–5.1 μm)**: I found general statements that
  this band exists and is used elsewhere (protostars, AGN), but **no PN-
  specific detection** of the fundamental (Δv=1) CO band in this range —
  only the first-overtone bandhead noted above, at ~2.3–2.5 μm, in a
  proto-PN.

**Bottom line for item 4/ices**: within 0.75–5.0 μm specifically, I can
confirm PAH (§5 above) and H₂ 1–0 S(1)/2–1 S(1) as real, PN-detected
molecular features; I cannot confirm any H₂O/CO₂/CO ice or gas band inside
this wavelength window for any PN from the sources I was able to read — the
real ice detections I found are all at wavelengths >14 μm, outside SPHEREx's
range.

---

## 6. Atlas/survey papers used

| Paper | Range | What it covers | Read status |
|---|---|---|---|
| Rudy, Lynch, Mazuk, Puetter & Dearborn (2001, AJ 121, 362) — "The Near-Infrared Spectrum of the Planetary Nebula IC 5117" | 0.8–2.5 μm | He II, [S III], [S II], [N I], H₂ in one PN | Abstract only (ResearchGate 403'd the full PDF; ADS abstract fetch also failed to return content) |
| Hora, Latter & Deutsch (1999, ApJS 124, 195, arXiv:astro-ph/9904202) — "Investigating the NIR Properties of PNe II: Medium-Resolution Spectra" | 1.0–2.5 μm, R≈700, 41 PNe | H I, He I, [Fe II], H₂, CO bandhead, C₂, CN across a 4-way morphological/spectral classification | **Read in full** (8 pages) |
| Ohsawa, Onaka, Sakon, Matsuura & Kaneda (2016, AJ 151, 93, arXiv:1604.00884) — "AKARI/IRC Near-Infrared Spectral Atlas of Galactic PNe" | 2.5–5.0 μm, R≈100, 72 PNe | H recombination lines + 3.3–3.5 μm hydrocarbon feature | Abstract only — could not get full-text line list |
| Bernard-Salas, Pottasch, Beintema & Wesselius (2001, A&A 367, 949) — "The ISO-SWS spectrum of planetary nebula NGC 7027" | 2.38–45.2 μm | Fine-structure lines, 9 H₂ lines, PAH bands (3.3, 6.2, 7.7, 11.3 μm) | Search-summary only |
| Mata, Ramos-Larios, Guerrero et al. (2016, MNRAS 459, 841, arXiv:1603.06667) — "Spitzer mid-infrared spectroscopic observations of planetary nebulae" | 5.2–14.5 μm, 157 compact Galactic PNe | [Ar III], [S IV], [Ne II] (most common), H₂ 0–0 S(2)–S(7), PAH in a few | Search-summary only; **wavelength range is entirely ≥5.2 μm, so strictly outside this task's 0.75–5.0 μm window** |
| Casassus, Roche & Barlow (2000, MNRAS 314, 657) — coronal-line regions of NGC 6302/6537, 3–13 μm | 3–13 μm | Coronal lines incl. [Si IX] 3.934 μm | Abstract only |
| Jones, Álvarez-Márquez, Sloan et al. (2023, MNRAS 523, 2519) — JWST MIRI MRS observations of PN SMP LMC 058 | 4.9–27.9 μm | 51 lines: H I Pfund/Humphreys series, [Fe II], [Ar II/III], [Ne II/III], [S III/IV], [Cl II], [K IV], [Na III], H₂(0,0) S lines | **Read in full** (6 pages incl. Table 1); confirms vacuum convention for H I λ_lab; **range only barely overlaps this task's window (4.9–5.0 μm)** |
| JWST/MIRI NGC 6302 paper (2025/2026, arXiv:2508.19332) | ≥4.9 μm | 66 H I, 54 He II, 8 H₂, ~50 forbidden lines total | Read in full for the <5 μm tail only: 3 lines total (He II 28-13 at 4.9088, H I 23-7 at 4.9237, H₂ 1-1 S(9) at 4.9541 μm) — **essentially no overlap** with 0.75–5.0 μm |
| Smith & McLean (2007, ApJ, arXiv:0712.0812) — SOFIA/FLITECAM 3.3 μm PAH survey of PNe | 3.0–3.4 μm | PAH 3.3 μm profile + Pfund-δ/ε, He II in a subset of 20 objects | **Read in full** (8 pages) |
| Akras, Monteiro, Walsh et al. (2024, A&A, arXiv:2409.00371) — "Detection of the [C I] λ8727 emission line... NGC 7009" | 0.75–0.98 μm (MUSE) | [C I] λ8727, λλ9824/9850 in NGC 7009 + literature compilation of ~40 other PNe (Table 1) | **Read in full** (3 pages) |

I did **not** find a dedicated JWST NIRSpec (0.6–5.3 μm) PN spectroscopic
atlas paper — the JWST PN papers I found (NGC 6302, SMP LMC 058, NGC 6720)
are all MIRI-only or MIRI-dominated, starting at 4.9 μm, so JWST currently
contributes almost nothing to the sub-4.9 μm part of this line list. This
should be stated plainly rather than implying JWST coverage that doesn't
exist yet for this specific band.

---

## 7. Comparison: repo's current values vs. what this research found

From `examples/fit_pn_lines.py::LINE_LIST` and `src/ssiaat/model/sed.py::get_pah()`.

| Repo entry | Repo value (μm) | Literature value | Verdict |
|---|---|---|---|
| Pa-γ | 1.09411 | vacuum 1.09412 | **Confirmed** — matches vacuum to 5 sig figs (Δ=0.00001) |
| Pa-β | 1.28181 | vacuum 1.28217 / air 1.28182 | **Convention mismatch, not confirmed as vacuum** — matches the *air* value almost exactly (Δ=0.00001 from air, Δ=0.00036 from vacuum). Internally inconsistent with Pa-γ/Br-δ/Br-γ/Br-β below, which all match vacuum. |
| Pa-α | 1.87510 | vacuum 1.87563 / air 1.87512 | **Convention mismatch, not confirmed as vacuum** — matches *air* (Δ=0.00002 from air, Δ=0.00053 from vacuum), same inconsistency as Pa-β |
| Br-δ | 1.94509 | vacuum 1.94510 / air 1.94456 | **Confirmed** — matches vacuum (Δ=0.00001), *not* air (Δ=0.00053) |
| Br-γ | 2.16612 | vacuum 2.16613 | **Confirmed** — Δ=0.00001 |
| Br-β | 2.62587 | vacuum 2.62588 | **Confirmed** — Δ=0.00001 |
| Br-α | 4.05116 | vacuum 4.05228 (NOT/IAC-table- and Rydberg-confirmed) | **Likely incorrect.** Off from the well-cross-checked vacuum value by 0.00112 μm (~83 km s⁻¹ equivalent) — a real, non-trivial offset for line-center fitting, unlike every other above-2 μm repo line (Br-γ, Br-β, Pf-β below), which all match vacuum to Δ≤0.00002. Recommend correcting to **4.05228 μm**. |
| Pf-β | 4.65378 | vacuum 4.65379 | **Confirmed** — Δ=0.00001 |
| He I 1.083 | 1.08303 | ~1.08303 (blended air-triplet value quoted across multiple sources) | **Confirmed**, but see §3 caveat that this is a not-fully-independently-verified single-source-family number, and is physically a blend of 3 fine-structure components (negligible for SPHEREx resolution) |
| He I 2.059 | 2.05869 | 2.0587 (UKIRT compilation + Hora et al. 1999) | **Confirmed** against 2 sources |
| [S III] 0.9069 | 0.90686 | 0.90686–0.9069 (air) across sources | **Confirmed** |
| [S III] 0.9531 | 0.95306 | 0.95306–0.9532 (air) across sources | **Confirmed**, ~1 Å source-to-source spread is normal rounding, not a discrepancy |
| PAH aromatic, `get_pah()` narrow component | center 3.2932 μm (midpoint of 3.2481–3.3382), half-width 0.0451 μm | Smith & McLean (2007): PNe are Class A3.3, center 3.288–3.290 μm, Gaussian FWHM 0.040–0.042 μm | **Not a clean match.** The repo's center (3.293 μm) sits ~0.003–0.005 μm redward of the literature's PN-specific (Class A3.3) center, and is instead close to the *Class B3.3* center (3.293/3.297 μm) — which the literature explicitly associates with H II regions/reflection nebulae, not PNe. Also, `hflattop`'s flat-top width (0.090 μm across the two edges) isn't directly comparable to a Gaussian FWHM (0.040 μm), since the shapes differ; a literature-matched re-tune of this component's center is worth doing. |
| PAH aliphatic, `get_pah()` broad component | 3.32–3.52 μm (center 3.42, half-width 0.10) | General literature: aliphatic feature "3.4–3.6 μm" | **Unable to verify precisely.** No PN-specific measurement of this component's exact boundaries was found (only a generic ISM/PPN aliphatic-feature wavelength range); the repo's range is broadly consistent with the general literature range but I found nothing to confirm or refute the specific 3.32/3.52 μm edges themselves. |

---

## 8. Candidate additions for the fitter (literature-supported, not yet in `LINE_LIST`)

In rough order of how confidently "common in PNe" each is supported by the
sources actually read in this session:

1. **[C I] 0.8727, 0.9824, 0.9850 μm** — confirmed common (Akras et al. 2024
   Table 1, ~40 PNe including IC 5117; Danziger & Goad 1973; Liu et al.
   1995). Falls in band 1 alongside the existing [S III] doublet; 0.9824/
   0.9850 sit close to [S III] 0.9531's expected FWHM tail (per the existing
   docstring's own note about the 0.94–0.96 μm cluster) — worth checking for
   blending before adding.
2. **H₂ 1–0 S(1) 2.1218 μm** — confirmed as the most commonly detected H₂
   line in PN near-IR spectra (Hora et al. 1999 and others); sits in band 3,
   near the repo's own noted unmodeled residual at "band 3 near 2.15–2.18 μm"
   (close but not coincident — worth checking if this residual is H₂ 1–0
   S(1) shifted, or something else).
3. **He I 1.7002 μm** and **He I 2.1126/2.1138 μm** — single-source
   confirmation each (Hora et al. 1999 for the former; a WebSearch-only
   summary for the latter, not independently re-verified) — lower priority
   than #1/#2 given the weaker sourcing, but real, cited PNe lines.
4. **[Fe II] 1.257, 1.644 μm** — confirmed but not universal: these are
   shock/low-ionization-structure (LIS) tracers present in a subset of PNe,
   not a core recombination/forbidden-line feature expected in every
   stacked PN. Worth adding as an optional/flagged pair given the SPHEREx
   stack pools 308 PNe, some fraction of which likely have LISs.
5. **Coronal lines [Si VI] 1.964, [Si VII] 2.48, [Si IX] 3.934 μm** —
   confirmed as real PN features (Casassus, Roche & Barlow 2000; Dinerstein
   review) but explicitly rare, requiring an unusually hot central star.
   Low priority for a "core" list; worth a one-line mention in the fitter's
   docstring as a known possibility rather than an active line to fit,
   unless the stacked spectrum shows a specific residual motivating it.
6. **Pfund-δ 3.29700, Pfund-ε 3.03921 μm** — real H I lines (confirmed via
   Rydberg formula + directly observed sitting on top of the PAH 3.3 μm
   feature in Smith & McLean 2007's NGC 7027 spectrum). Given the repo's
   `fit_pn_lines.py` already treats the PAH complex specially (fixed-shape
   `extra_models`, not a `LineSpec`), Pfund-δ in particular is a real,
   confirmed line sitting inside that exact wavelength window and is
   presently neither fit as a line nor accounted for in the PAH shape —
   worth flagging for whoever tunes band 4 next.

**Not recommended without further work:** [N I] near 0.87 μm (§4,
unverifiable, possibly confused with [C I] 0.8727); CO₂/H₂O/CO ice or gas
bands (§5, no PN detection found inside 0.75–5.0 μm); Humphreys-series H I
members below Hu-δ (§2, theoretical/Rydberg-only, no confirmed PN
detection).

---

## References

- NIST Atomic Spectra Database (ASD), Lines Query form and stated
  air/vacuum convention: https://physics.nist.gov/PhysRefData/ASD/lines_form.html,
  https://physics.nist.gov/cgi-bin/ASD/lines1.pl
- Hummer, D. G., & Storey, P. J. 1987, MNRAS, 224, 801, "Recombination-line
  intensities for hydrogenic ions – I. Case B calculations for H I and He II"
- Storey, P. J., & Hummer, D. G. 1995, VizieR Online Data Catalog
  1995yCat.6064....0S, "Recombination line intensities for hydrogenic ions"
  (Case A/B)
- Storey, P. J., & Zeippen, C. J. 2000 (cited for the fixed [S III]
  λ9531/λ9069 intensity ratio; not independently read in this session,
  cited via a secondary source)
- CODATA 2018 fundamental physical constants (R_∞, mₑ/M_p), used for the
  Rydberg-formula H I wavelength calculation in §2
- Peck, E. R., & Reeder, K. 1972, J. Opt. Soc. Am., 62, 958 (standard-air
  dispersion formula, as documented by NIST ASD for its air-wavelength
  conversion)
- NOT (Nordic Optical Telescope) NOTCam Reference Info: Paschen, Brackett,
  Pfund series calibration-line pages,
  https://www.not.iac.es/instruments/notcam/ReferenceInfo/
- UKIRT (University of Hawaii IfA) astronomical/calibration line pages:
  Helium Lines, Paschen/Brackett series, https://about.ifa.hawaii.edu/ukirt/calibration-and-standards/spectroscopic-calibration/astronomical-and-calibration-lines/
- Rudy, R. J., Lynch, D. K., Mazuk, S., Puetter, R. C., & Dearborn, D. S. P.
  2001, AJ, 121, 362, "The Near-Infrared Spectrum of the Planetary Nebula
  IC 5117" (ADS: 2001AJ....121..362R)
- Hora, J. L., Latter, W. B., & Deutsch, L. K. 1999, ApJS, 124, 195,
  "Investigating the Near-Infrared Properties of Planetary Nebulae. II.
  Medium Resolution Spectra" (arXiv:astro-ph/9904202)
- Ohsawa, R., Onaka, T., Sakon, I., Matsuura, M., & Kaneda, H. 2016, AJ,
  151, 93, "AKARI/IRC Near-Infrared Spectral Atlas of Galactic Planetary
  Nebulae" (arXiv:1604.00884)
- Bernard-Salas, J., Pottasch, S. R., Beintema, D. A., & Wesselius, P. R.
  2001, A&A, 367, 949, "The ISO-SWS spectrum of planetary nebula NGC 7027"
- Mata, H., Ramos-Larios, G., Guerrero, M. A., Nigoche-Netro, A., Toalá,
  J. A., Fang, X., Rubio, G., Kemp, S. N., Navarro, S. G., & Corral, L. J.
  2016, MNRAS, 459, 841, "Spitzer mid-infrared spectroscopic observations
  of planetary nebulae" (arXiv:1603.06667)
- Casassus, S., Roche, P. F., & Barlow, M. J. 2000, MNRAS, 314, 657,
  coronal-line regions of planetary nebulae NGC 6302 and 6537, 3–13 μm
  grating and echelle spectroscopy
- Jones, O. C., Álvarez-Márquez, J., Sloan, G. C., Kavanagh, P. J.,
  Argyriou, I., Law, D. R., et al. 2023, MNRAS, 523, 2519, "Observations of
  the planetary nebula SMP LMC 058 with the JWST MIRI medium resolution
  spectrometer"
- JWST/MIRI view of the planetary nebula NGC 6302 I. (2025/2026, MNRAS 542,
  1287, arXiv:2508.19332)
- Smith, E. C. D., & McLean, I. S. 2007, ApJ (accepted; arXiv:0712.0812),
  "A Survey of 3.3 μm PAH Emission in Planetary Nebulae"
- Tokunaga, A. T., Sellgren, K., Smith, R. G., Nagata, T., Sakata, A., &
  Nakada, Y. 1991, ApJ, 380, 452 (3.3 μm PAH classification, cited via
  Smith & McLean 2007)
- van Diedenhoven, B., Peeters, E., Van Kerckhoven, C., Hony, S., Hudgins,
  D. M., Allamandola, L. J., & Tielens, A. G. G. M. 2004, ApJ, 611, 928
  (Class A/B 3.3 μm PAH classification, cited via Smith & McLean 2007)
- Roche, P. F., Lucas, P. W., Hoare, M. G., Aitken, D. K., & Smith, C. H.
  1996, MNRAS, 280, 924 (C/O threshold for PAH emission in PNe, cited via
  Smith & McLean 2007)
- Cohen & Barlow 2005, MNRAS, 362, 1199 (7/8 μm PAH C/O threshold, cited via
  Smith & McLean 2007)
- Akras, S., Monteiro, H., Walsh, J. R., Konstantinou, L., Gonçalves, D. R.,
  Garcia–Rojas, J., Boumis, P., & Aleman, I. 2024, A&A (arXiv:2409.00371),
  "Detection of the [C I] λ8727 emission line: Low-ionization structures in
  NGC 7009"
- Danziger, I. J., & Goad, L. 1973 (first [C I] λ8727 detection in NGC 7027,
  cited via Akras et al. 2024 — not independently read)
- Liu, X.-W., et al. 1995 ([C I] detections in 4 additional PNe, cited via
  Akras et al. 2024 — not independently read)
- Akras et al. 2024, A&A 689, A70, "[Fe II] 1.644 μm imaging survey of
  planetary nebulae with low-ionisation structures" (abstract/summary read
  via search, not the full paper)
- Dinerstein, H. L., "Infrared Emission Lines as Probes of Gaseous Nebulae"
  (review), NED: https://ned.ipac.caltech.edu/level5/March01/Dinerstein2/Diner3.html
- Cohen, M., Barlow, M. J., Sylvester, R. J., Liu, X.-W., Cox, P., Lim, T.,
  Schmitt, B., & Speck, A. K. 1999, arXiv:astro-ph/9901126, "Water Ice,
  Silicate and PAH Emission Features in the ISO Spectrum of the Carbon-rich
  Planetary Nebula CPD−56°8032"
- "Detection of CO₂ ice in the planetary nebula NGC 6302" 2026, A&A
  (arXiv:2602.22366) — found via search, not independently read in full
- Storey, P. J. & Hummer, D. G. relative-intensity data as re-used/cited by
  the NOT/IAC and UKIRT calibration-line web pages above (secondary use)

**Sources consulted but not usable** (dead-end for the record): ResearchGate
page for Rudy et al. (2001) IC 5117 (HTTP 403); NASA ADS abstract fetch for
the same paper (returned empty); NIST ASD raw HTML line-query output (mis-
extracted by WebFetch's summarizer on first attempts — see the digit-
transposition note in the intro; worked around via the Rydberg-formula
calculation plus independent cross-checks instead of trusting that
extraction).

---

## 9. Targeted search: unidentified band-6 feature near 4.47 μm

Research note extending this file to a specific open question from
`examples/fit_pn_lines.py`: the 308-PN SPHEREx stack shows a real,
~15σ, narrow (instrumental FWHM ≈0.034 μm) emission feature at
**≈4.4728–4.475 μm (vacuum)**, tracked there as `LineSpec "Unid. 4.47um
band6"`. H I (full Humphreys series via Rydberg–Ritz) and He II (Z=2
hydrogenic Rydberg–Ritz series) were already computed and ruled out
earlier in this same session — nearest members Hu-ζ (4.3765 μm) / Hu-ε
(4.6725 μm) for H I and He II 8→7 (4.7635 μm, matching a *separate*
already-identified ~4.76 μm bump) — both >0.09 μm (>2.5 FWHM) away; not
re-derived here, only cross-checked against NIST ASD where noted below.

### 9.1 Coronal/high-ionization candidates, checked directly against NIST ASD

Each ion below was queried directly against the NIST ASD Lines Query
(https://physics.nist.gov/cgi-bin/ASD/lines1.pl, live query in this
session, vacuum wavelengths, 4–5 μm window unless noted) rather than
taken from a secondary summary.

| Ion | NIST ASD λ (vac, μm) | Transition | Offset from 4.4728–4.475 μm | Verdict |
|---|---|---|---|---|
| **[Mg IV]** | **4.488** (ground term ²P₃/₂–²P₁/₂, M1; also a much weaker E2 at the same λ) | forbidden, ground-term fine-structure | **+0.013 to +0.015 μm (≈0.4×FWHM), redward** | **Closest real candidate found — but not a precise match** (see §9.5) |
| [Ar VI] | 4.5292 (²P₁/₂–²P₃/₂, M1) | forbidden | +0.054 to +0.056 μm (>1.5×FWHM) | **Checked and ruled out** |
| [Mg V] | no line in 4–5 μm matches; nearest catalogued lines are 4.0005 (E2, very weak), 4.134, 4.264 μm. The real diagnostic ground-term line (³P₂–³P₁, M1) is at **5.608 μm** (per Casassus et al. 2000, §9.5), far outside this window | forbidden (ground-term line outside 4–5 μm) | ≥0.21 μm (≥6×FWHM) for anything in-window | **Checked and ruled out** |
| [Ca V] | 4.1585 (M1, plus a much weaker E2 at the same λ) | forbidden | −0.31 to −0.32 μm (≥9×FWHM) | **Checked and ruled out** |
| [Na III] | **no lines returned by NIST ASD anywhere in 4–5 μm** ("No lines are available in ASD with the parameters selected") | — | n/a | **Not tabulated in this window — not a candidate** |
| [Al V] | **no lines returned by NIST ASD anywhere in 4–5 μm.** Its real ground-term M1 line is at 2.883 μm (Casassus et al. 2000, matches NIST-style value in their Table A1) | forbidden (ground-term line far outside window) | n/a | **Not tabulated in this window — not a candidate** |
| [Si IX] | **no lines returned by NIST ASD anywhere in 4–5 μm.** Its one known IR line is 3.934 μm (already documented in §4 of this file, confirmed via Casassus et al. 2000 + Dinerstein review) | forbidden (known line far outside window) | n/a | **Not tabulated in this window — not a candidate** |

### 9.2 Comprehensive all-species NIST ASD scan of 4.45–4.50 μm

To catch anything not on the task's candidate list, NIST ASD was queried
with **no species restriction** ("Spectra" field left blank) across the
full 4.45–4.50 μm window — this returns every line from every element/ion
NIST ASD has catalogued in that range, not just the coronal ions above.
The query returned 37 lines total; the ones close enough to be worth
listing (full offset from the 4.4728–4.475 μm centroid) are:

| Species | λ (vac, μm) | Type | Offset | Assessment |
|---|---|---|---|---|
| Si I | 4.47318 | allowed (E1), Aki=8.58×10³ s⁻¹ | **inside** the quoted 4.4728–4.475 μm range | Wavelength match is essentially exact, but this is a *permitted* transition of *neutral* silicon — no known nebular-emission mechanism and no PN detection found in any source read in this session. **Precise wavelength match; astrophysically implausible / unconfirmed carrier** |
| Xe I | 4.47324 (×2, blended) | not classified in ASD output | **inside** the quoted range | Same verdict as Si I: neutral noble-gas atomic line, no plausible PN nebular-emission mechanism, no literature PN detection found. **Precise wavelength match; astrophysically implausible / unconfirmed carrier** |
| Fe II | 4.4790036 | not classified/tabulated (no Aki in ASD) | +0.004 to +0.006 μm (<0.2×FWHM) | Real ion, but NIST ASD does not classify this particular transition's strength, and no PN paper reporting emission at 4.479 μm was found. **Could not verify — inconclusive, not a confirmed candidate** |
| Fe XIV | 4.49 (E2/M1, both listed) | forbidden | +0.015 to +0.017 μm | Requires stripping to Fe¹³⁺ (ionization energies far beyond even the "record-setting" [Si IX] IP=303 eV noted in Casassus et al. 2000 as the highest confirmed PN coronal ionization state). **Checked and ruled out on physical grounds** despite the wavelength being close |
| Mg IX | 4.464 (allowed) | permitted | −0.009 to −0.012 μm | Requires ~265 eV to produce from Mg VIII; no PN central star is known to be hot enough. **Checked and ruled out on physical grounds** |
| Ac I, Th I, Th II, Na II, C I, O IV, Li I, S I, S II, Cd II | 4.46–4.50 (various) | mostly allowed/lab lines | 0.005–0.04 μm | Actinium/thorium entries are laboratory calibration lines with no astrophysical relevance; Na II/C I/O IV/Li I/S I/S II/Cd II are all either far enough off (≥0.005–0.04 μm, i.e. comparable to or larger than the Mg IV offset) or have no reported PN mid-IR detection. **Checked and ruled out / not credible candidates** |

None of these is a forbidden ground-term "coronal-type" line the way
[Mg IV]/[Ar VI]/[Mg V]/[Ca V]/[Si IX] are, and none has been reported as a
detected PN emission feature in any primary source read in this session.

### 9.3 H₂ and other molecular/dust features near 4.47 μm

H₂ pure-rotational (0–0) S-branch wavelengths, from the MPE/ISO H₂ line
list (https://www.mpe.mpg.de/ir/ISO/linelists/H₂.html — an ISO-heritage
compilation, read directly in this session): **S(9) = 4.69466 μm**,
**S(10) = 4.40997 μm** — the two S-branch members bracketing 4.47 μm, and
both **>0.06 μm (>1.7×FWHM) away**; ruled out, consistent with §5's
finding that the well-established, commonly-detected PN H₂ lines
(1–0 S(1) 2.1218 μm, 2–1 S(1) 2.2477 μm) are nowhere near this window.
Higher-v vibration–rotation H₂ lines that happen to fall closer in
wavelength — e.g. 7–6 Q(13) 4.42179 μm, 9–8 Q(3) 4.43663 μm, 2–1 O(8)
4.43739 μm, all from the same MPE list — are still ≥0.03 μm (≥0.9×FWHM)
off *and* require v′=7–9 vibrational excitation that is not expected in
an ordinary (non-shocked) PN, let alone consistently across a stack of
308 objects; **checked and ruled out**.

No PN-specific detection of the CO fundamental (Δv=1, ~4.3–5.1 μm) band or
the CO₂ ice 4.27 μm stretching mode was found in this targeted search
either, consistent with §5's conclusion for the whole 0.75–5.0 μm window:
**no PN-specific molecular/ice feature inside 0.75–5.0 μm was confirmed
anywhere in this file's research, including specifically at 4.47 μm.**

### 9.4 Survey/atlas corroboration: does any dedicated PN mid-IR paper flag anything at 4.47 μm?

- **Ohsawa, Onaka, Sakon, Matsuura & Kaneda (2016), AJ 151, 93
  (arXiv:1604.00884)** — the AKARI/IRC 2.5–5.0 μm, R≈100 atlas of 72
  Galactic PNe already listed in §6 as "abstract only" — was **read in
  full in this session** (their Tables 1–6, Figures 4–7). Their spectral
  resolution (FWHM of the spectral PSF ≈0.035–0.06 μm) is directly
  comparable to SPHEREx's 0.034 μm FWHM at this wavelength, making this
  the single most relevant comparison survey for this question. Their
  fixed fitting line list (Table 3) includes exactly two fine-structure
  features in this region: **[Mg IV] at 4.487 μm** and **[Ar VI] at
  4.530 μm** — both independently detected across multiple individual
  objects in their sample (Table 5 lists per-object [Mg IV] fluxes for
  PNe 002.0−13.4, 038.2+12.0, and others). **No additional or
  unidentified feature near 4.47 μm is mentioned anywhere in their text,
  tables, or example spectra** (Fig. 4's PNG 074.5+02.1 panel shows only
  [Mg IV]/[Ar VI] labeled between 4.3 and 4.8 μm). This is a real,
  useful negative result: the one published PN survey with matching
  resolution and comparable spectral coverage does not flag a distinct
  line at 4.47 μm.
- **Smith, Logsdon, McLean et al. (2020), ApJ (arXiv:2008.13635)**,
  "A Survey of 3 to 5.4 μm Emission From Planetary Nebulae using
  SOFIA/FLITECAM" (already cited in §5 for its PAH work) — **read in full
  in this session**. Confirms [Mg IV] independently: "IC 5117 showed
  [Mg IV] emission" with flux 4.1±0.18×10⁻¹⁵ W/m² (their Table 3),
  labeled at ≈4.49 μm in their Figure 2 — consistent with the NIST/AKARI
  value to the precision quoted. No feature at 4.47 μm specifically is
  reported for any of their three targets (BD+30°3639, IC 5117,
  PNG 093.9−00.1); their 4.2–4.4 μm range was in any case unobservable
  ("opaque even at SOFIA altitudes"), so this paper cannot rule anything
  in or out for the low-wavelength edge of the target window, only for
  4.4–4.475–ish upward.
- **Casassus, Roche & Barlow (2000), MNRAS 314, 657** (already cited in
  §4/§8 for [Si VI]/[Si IX]) — **re-read in full in this session**,
  including their Appendix A wavelength table. Covers [Mg VIII] 3.028,
  [Mg VII] 5.502, [Al V] 2.883/[Al VI] 3.659, [Si VI] 1.963/[Si VII]
  2.48/[Si IX] 3.934, [Ar V] 7.91+13.1/[Ar VI] 4.528, [Na VI] 8.64/
  [Na VII] 4.69, [K VII] 3.19 — **no line anywhere near 4.47 μm** other
  than [Ar VI] (already ruled out above, §9.1).
- **Bernard-Salas, Pottasch, Beintema & Wesselius (2001), A&A 367, 949**
  — the dedicated ISO-SWS NGC 7027 line-list paper (2.38–45.2 μm, so it
  does nominally cover 4.47 μm) — **could not be read in full in this
  session**: the A&A PDF returned HTTP 403, and the line-list tables
  (Tables 1/3/4, containing the actual per-wavelength identifications)
  are noted in the paper itself as CDS/electronic-only and were not
  retrieved. This is the same limitation already flagged for this paper
  in §6 ("search-summary only"); **explicitly flagged here as "could not
  verify"** rather than assumed to contain or not contain a 4.47 μm
  entry.
- **Zhang, Liu, Luo, Péquignot & Barlow (2005), arXiv:astro-ph/0507155**,
  "Integrated spectrum of the planetary nebula NGC 7027" — read in full;
  their deep line list is **optical-only (3310–9160 Å)** and does not
  cover 4.47 μm at all. Consulted, not applicable.
- No dedicated JWST NIRSpec (0.6–5.3 μm) PN spectroscopic atlas covering
  4.47 μm was found in this targeted search either, consistent with §6's
  standing conclusion that current JWST PN papers are MIRI-dominated and
  start at ≥4.9 μm.

### 9.5 Bottom line

**No confirmed identification was found for the ≈4.4728–4.475 μm
feature.** [Mg IV] at 4.487–4.488 μm (NIST ASD; independently confirmed
as a real, detected PN line in both a single-object near-IR survey
— Smith et al. 2020, IC 5117 — and a 72-object atlas — Ohsawa et al.
2016) is by a wide margin the closest known forbidden/coronal line to
this wavelength, and it is astrophysically plausible for a 308-PN stack
(its 80 eV ionization requirement is comparable to [Ar VI]'s 75 eV,
both attainable in a modest fraction of moderately hot PN central
stars). But its wavelength sits **0.012–0.015 μm (≈35–44% of the
0.034 μm instrumental FWHM) redward** of the quoted centroid — a real,
non-negligible offset, larger than the wavelength-calibration accuracy
implied by how tightly the H I/He II Rydberg-Ritz cross-checks matched
earlier in this same investigation. This should be treated as **the
closest known candidate, not a confirmed match** — worth a follow-up
check of SPHEREx's own band-6 wavelength-solution accuracy specifically
at 4.47–4.49 μm before going further, but not asserted here as an
identification.

Every other coronal ion named in the task is either definitively ruled
out by direct NIST ASD wavelengths ([Ar VI] 4.529 μm, [Mg V]'s real line
at 5.608 μm, [Ca V] 4.1585 μm — all ≥1.5 FWHM away) or has **no line
tabulated in NIST ASD anywhere in the 4–5 μm window at all** ([Na III],
[Al V], [Si IX]). A comprehensive, species-blind NIST ASD scan of the
entire 4.45–4.50 μm window surfaces exactly two lines whose wavelength
sits *inside* the observed centroid range (Si I 4.47318 μm, Xe I
4.47324 μm) — but both are permitted transitions of neutral, easily
photoionized atoms, with no known nebular-emission mechanism and no PN
literature detection found; not credible candidates on the evidence
available. No H₂ line (nearest: 0–0 S(10) at 4.40997 μm, >1.7 FWHM away),
no other molecular transition, and no PN-specific ice/CO/CO₂ feature
lands within even one FWHM of the observed wavelength. The one published
near-IR PN survey with genuinely comparable spectral resolution (AKARI/
IRC, 72 PNe) reports no feature at all near 4.47 μm beyond [Mg IV]/
[Ar VI]. **This is a clean "not found" result** for a precise
identification; [Mg IV] is flagged only as the nearest known candidate
for whoever investigates this further, not as a resolved answer.

### References added for §9

- NIST Atomic Spectra Database (ASD), Lines Query, direct queries for
  Mg IV, Mg V, Ca V, Na III, Al V, Si IX, Ar VI, Fe II, Si I, and an
  all-species scan of 4.45–4.50 μm, all run in this session:
  https://physics.nist.gov/cgi-bin/ASD/lines1.pl (citing, per NIST's own
  attribution for these entries, Martin & Zalubas 1980 and Kaufman &
  Martin 1991 for Mg-ion energy levels/classifications, and Saloman 2010
  for Ar VI)
- Ohsawa, R., Onaka, T., Sakon, I., Matsuura, M., & Kaneda, H. 2016, AJ,
  151, 93, "AKARI/IRC Near-Infrared Spectral Atlas of Galactic Planetary
  Nebulae" (arXiv:1604.00884) — already listed in the main References
  list above as abstract-only; **upgraded here to read-in-full** for
  this section's purposes
- Smith, E. C., Logsdon, S. E., McLean, I. S., Fletcher, E., Vacca,
  W. D., Becklin, E. E., Shenoy, S., Savage, M., & Hamilton, R. T. 2020,
  ApJ, "A Survey of 3 to 5.4 μm Emission From Planetary Nebulae using
  SOFIA/FLITECAM" (arXiv:2008.13635)
- Pereira-Santaella, M., et al. 2024, A&A, 685, L13, "Extended
  high-ionization [Mg IV] emission tracing widespread shocks in
  starbursts seen by JWST/NIRSpec" (arXiv:2404.16641) — extragalactic,
  cited only as independent cross-confirmation of the [Mg IV] 4.487 μm
  laboratory wavelength, not as PN evidence
  https://www.aanda.org/articles/aa/full_html/2024/05/aa49982-24/aa49982-24.html
- Zhang, Y., Liu, X.-W., Luo, S.-G., Péquignot, D., & Barlow, M. J. 2005,
  A&A, 442, 249, "Integrated spectrum of the planetary nebula NGC 7027"
  (arXiv:astro-ph/0507155) — consulted, optical-only, not applicable to
  this section's wavelength range
- MPE/ISO H₂ pure-rotational and vibration–rotation line list:
  https://www.mpe.mpg.de/ir/ISO/linelists/H2.html
- Kaufman, V., & Martin, W. C. 1991 (Mg-ion spectral classifications, as
  cited by NIST ASD — not independently read in this session)
- Martin, W. C., & Zalubas, R. 1980 (Mg-ion energy levels, as cited by
  NIST ASD — not independently read in this session)
- Saloman, E. B. 2010 (Ar-ion energy levels/wavelengths, as cited by
  NIST ASD for Ar VI — not independently read in this session)

**Sources consulted but not usable for §9** (dead-end for the record):
A&A's own PDF host for Bernard-Salas et al. (2001) — HTTP 403, same as
noted for the main body of this file; the paper's actual line-list
tables are CDS/electronic-only per the paper's own text and were not
retrieved by any route tried in this session.

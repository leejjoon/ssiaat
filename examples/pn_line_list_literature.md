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

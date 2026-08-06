"""MultiBandLineModel: tied-center, per-band-amplitude emission-line fits.

All synthetic -- no network, no dependency on the real pn_master.ecsv (the
example script that fits that file is examples/fit_pn_lines.py, not
tested here). Ground truth is built using the module's OWN
normalization/baseline helpers (`_normalized_profile`, `_baseline_vander`)
rather than a hand-rolled reference, so the noiseless test isolates
"does the optimizer recover known parameters" from "does a by-hand
reference implementation agree with the module's own convention".
"""
import numpy as np
import pytest

from ssiaat.bandpass_shape_lite import BandpassShapeLite
from ssiaat.model.multiband_lines import (
    BandSpectrum,
    LineSpec,
    MultiBandLineModel,
    _baseline_vander,
    _normalized_profile,
    _profile_shape,
)

SHAPE = BandpassShapeLite()


def _log_grid(w0, w1, n):
    ratio = (w1 / w0) ** (1 / (n - 1))
    return w0 * ratio ** np.arange(n)


# Real band-1/band-2 wavelength ranges and point counts, from
# pn_master.ecsv (method="A_pool"), so the synthetic grids match the
# actual log-spacing/resolution this module was designed against.
BAND1_GRID = _log_grid(0.7352704, 1.1213731914357052, 254)
BAND2_GRID = _log_grid(1.0877753517973583, 1.6636861918458434, 255)


def _true_value(band, wavelength, lines_here, baseline_coeffs):
    """sum(amp * normalized_profile(center)) + chebyshev(baseline_coeffs),
    using the module's own helpers -- the noiseless ground truth."""
    value = np.zeros_like(wavelength)
    for center, amp in lines_here:
        shape_kwargs, _, _ = _profile_shape(SHAPE, center, band, allow_extrapolated=True)
        profile = _normalized_profile(center, **shape_kwargs)
        value = value + amp * profile(wavelength)
    span = (wavelength.min(), wavelength.max())
    vander = _baseline_vander(wavelength, span, len(baseline_coeffs) - 1)
    return value + vander @ np.asarray(baseline_coeffs)


# --------------------------------------------------------------------------
# Test 1: noiseless exact recovery
# --------------------------------------------------------------------------

TRUE_CENTER_A = 1.0900   # inside both band 1 and band 2
TRUE_CENTER_B = 1.3500   # inside band 2 only
AMP_A_BAND1 = 3.0
AMP_A_BAND2 = 2.5        # deliberately different from band 1 -- the whole point
AMP_B_BAND2 = 1.5
BASELINE1 = [1.0, 0.05, -0.02]
BASELINE2 = [1.0, -0.03, 0.01]


@pytest.fixture
def noiseless_setup():
    val1 = _true_value(1, BAND1_GRID, [(TRUE_CENTER_A, AMP_A_BAND1)], BASELINE1)
    val2 = _true_value(2, BAND2_GRID, [(TRUE_CENTER_A, AMP_A_BAND2),
                                       (TRUE_CENTER_B, AMP_B_BAND2)], BASELINE2)
    err1 = np.full_like(BAND1_GRID, 0.05)
    err2 = np.full_like(BAND2_GRID, 0.05)
    lines = [LineSpec("A", TRUE_CENTER_A), LineSpec("B", TRUE_CENTER_B)]
    spectra = [
        BandSpectrum(1, BAND1_GRID, val1, err1, baseline_degree=len(BASELINE1) - 1),
        BandSpectrum(2, BAND2_GRID, val2, err2, baseline_degree=len(BASELINE2) - 1),
    ]
    return lines, spectra


def test_band_assignment_excludes_out_of_range_line(noiseless_setup):
    lines, spectra = noiseless_setup
    model = MultiBandLineModel(lines, spectra)
    assignment = model.assignment()

    a_bands = sorted(assignment.loc[(assignment.line == "A") & assignment.assigned, "band"])
    b_bands = sorted(assignment.loc[(assignment.line == "B") & assignment.assigned, "band"])
    assert a_bands == [1, 2]
    assert b_bands == [2]  # B is far outside band 1's range -- must not be assigned there


def test_noiseless_recovers_centers_and_per_band_amplitudes(noiseless_setup):
    lines, spectra = noiseless_setup
    model = MultiBandLineModel(lines, spectra)
    result = model.fit()

    np.testing.assert_allclose(result.centers["A"], TRUE_CENTER_A, atol=1e-4)
    np.testing.assert_allclose(result.centers["B"], TRUE_CENTER_B, atol=1e-4)

    amp = result.amplitudes
    a1 = amp.loc[(amp.line == "A") & (amp.band == 1), "amplitude"].iloc[0]
    a2 = amp.loc[(amp.line == "A") & (amp.band == 2), "amplitude"].iloc[0]
    b2 = amp.loc[(amp.line == "B") & (amp.band == 2), "amplitude"].iloc[0]
    np.testing.assert_allclose(a1, AMP_A_BAND1, rtol=1e-4)
    np.testing.assert_allclose(a2, AMP_A_BAND2, rtol=1e-4)
    np.testing.assert_allclose(b2, AMP_B_BAND2, rtol=1e-4)
    # The test an accidental global-amplitude bug would fail: the SAME
    # line's amplitude really is different band to band.
    assert abs(a1 - a2) > 0.1


def test_noiseless_recovers_baseline_coefficients(noiseless_setup):
    lines, spectra = noiseless_setup
    model = MultiBandLineModel(lines, spectra)
    result = model.fit()

    solved1 = result.band_solutions[1]
    solved2 = result.band_solutions[2]
    # last (baseline_degree+1) coefficients are the baseline, in order
    coeffs1 = solved1["coeffs"][-len(BASELINE1):]
    coeffs2 = solved2["coeffs"][-len(BASELINE2):]
    np.testing.assert_allclose(coeffs1, BASELINE1, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(coeffs2, BASELINE2, rtol=1e-4, atol=1e-6)


def test_model_curve_matches_data_noiseless(noiseless_setup):
    lines, spectra = noiseless_setup
    model = MultiBandLineModel(lines, spectra)
    result = model.fit()
    for spec in spectra:
        curve = result.model_curve(spec.band)
        np.testing.assert_allclose(curve, spec.value, atol=1e-4)


# --------------------------------------------------------------------------
# Test 2: noisy recovery, per-band offset, error calibration
# --------------------------------------------------------------------------

REST_CENTER = 1.0900
DELTA_BAND1 = -0.0010
DELTA_BAND2 = -0.0030
AMP_BAND1 = 8.0   # strong lines: this tests machinery correctness, not
AMP_BAND2 = 7.0   # marginal-detection behavior


def _correlated_noise(rng, n, scale, kernel_width=20):
    white = rng.normal(scale=scale, size=n)
    kernel = np.ones(kernel_width) / kernel_width
    smoothed = np.convolve(white, kernel, mode="same")
    # Convolving with a normalized kernel divides the per-point std by
    # sqrt(sum(kernel**2)) -- rescale back up so the result has the same
    # per-point std as `scale` (matching what `error` claims) while still
    # being correlated over ~kernel_width samples. This correlation, not
    # a variance mismatch, is what should inflate the matched-filter S/N
    # distribution's MAD relative to the formal (independent-noise)
    # expectation.
    return smoothed / np.sqrt(np.sum(kernel ** 2))


@pytest.fixture
def noisy_offset_setup():
    rng = np.random.default_rng(20260806)
    noise_scale = 0.05

    val1_clean = _true_value(1, BAND1_GRID, [(REST_CENTER + DELTA_BAND1, AMP_BAND1)],
                             BASELINE1)
    val2_clean = _true_value(2, BAND2_GRID, [(REST_CENTER + DELTA_BAND2, AMP_BAND2)],
                             BASELINE2)
    val1 = val1_clean + _correlated_noise(rng, len(BAND1_GRID), noise_scale)
    val2 = val2_clean + _correlated_noise(rng, len(BAND2_GRID), noise_scale)

    err1 = np.full_like(BAND1_GRID, noise_scale)
    err2 = np.full_like(BAND2_GRID, noise_scale)
    lines = [LineSpec("A", REST_CENTER)]
    spectra = [
        BandSpectrum(1, BAND1_GRID, val1, err1, baseline_degree=len(BASELINE1) - 1),
        BandSpectrum(2, BAND2_GRID, val2, err2, baseline_degree=len(BASELINE2) - 1),
    ]
    return lines, spectra


def test_wavelength_solution_mode_recovers_per_band_offset(noisy_offset_setup):
    lines, spectra = noisy_offset_setup
    model = MultiBandLineModel(lines, spectra, centers_mode="wavelength_solution")
    result = model.fit()

    np.testing.assert_allclose(result.deltas[1], DELTA_BAND1, atol=0.0015)
    np.testing.assert_allclose(result.deltas[2], DELTA_BAND2, atol=0.0015)
    np.testing.assert_allclose(result.amplitudes["amplitude"],
                               [AMP_BAND1, AMP_BAND2], rtol=0.1)


def test_free_mode_recovers_a_common_center_near_rest(noisy_offset_setup):
    lines, spectra = noisy_offset_setup
    model = MultiBandLineModel(lines, spectra, centers_mode="free")
    result = model.fit()
    # With two bands offset in opposite directions from rest by <=0.003um,
    # one shared free center should still land close to rest.
    np.testing.assert_allclose(result.centers["A"], REST_CENTER, atol=0.003)


def test_null_mad_detects_correlation_and_is_near_unity_for_white_noise():
    rng = np.random.default_rng(42)
    noise_scale = 0.05

    def build(noise_fn):
        val = _true_value(1, BAND1_GRID, [(REST_CENTER, AMP_BAND1)], BASELINE1)
        val = val + noise_fn()
        err = np.full_like(BAND1_GRID, noise_scale)
        lines = [LineSpec("A", REST_CENTER)]
        spectra = [BandSpectrum(1, BAND1_GRID, val, err, baseline_degree=len(BASELINE1) - 1)]
        return MultiBandLineModel(lines, spectra).fit()

    correlated = build(lambda: _correlated_noise(rng, len(BAND1_GRID), noise_scale))
    white = build(lambda: rng.normal(scale=noise_scale, size=len(BAND1_GRID)))

    # MAD on a modest sample has real seed-to-seed variance; checked across
    # several seeds the correlated case lands ~1.3-3.9 and white ~0.6-1.2,
    # so a clear qualitative gap (not a specific numeric target) is what's
    # robust to assert here.
    assert correlated.null_mad[1] > 1.5
    assert 0.7 < white.null_mad[1] < 1.5


# --------------------------------------------------------------------------
# Test 3: guardrails
# --------------------------------------------------------------------------

def _simple_spectra():
    val1 = _true_value(1, BAND1_GRID, [(0.90, 5.0)], [1.0])
    val2 = _true_value(2, BAND2_GRID, [(1.30, 5.0)], [1.0])
    err1 = np.full_like(BAND1_GRID, 0.05)
    err2 = np.full_like(BAND2_GRID, 0.05)
    return [
        BandSpectrum(1, BAND1_GRID, val1, err1, baseline_degree=0),
        BandSpectrum(2, BAND2_GRID, val2, err2, baseline_degree=0),
    ]


def test_line_outside_band_domain_flagged_and_excluded_by_default():
    spectra = _simple_spectra()
    lo, hi = BandpassShapeLite().SHAPE_CHEB_DOMAIN[1]
    lines = [LineSpec("out_of_domain", hi + 0.5)]
    model = MultiBandLineModel(lines, spectra)
    assignment = model.assignment()
    row = assignment[(assignment.line == "out_of_domain") & (assignment.band == 1)].iloc[0]
    assert row.shape_extrapolated
    assert not row.assigned


def test_line_outside_band_domain_included_when_allowed():
    spectra = _simple_spectra()
    lo, hi = BandpassShapeLite().SHAPE_CHEB_DOMAIN[1]
    lines = [LineSpec("out_of_domain", hi + 0.01)]  # just outside, small extrapolation
    model = MultiBandLineModel(lines, spectra, allow_extrapolated_shape=True)
    assignment = model.assignment()
    row = assignment[(assignment.line == "out_of_domain") & (assignment.band == 1)].iloc[0]
    assert row.shape_extrapolated


def test_wavelength_solution_raises_for_band_with_no_assigned_line():
    spectra = _simple_spectra()
    # Only assigned in band 1 -- band 2 has no line at all.
    lines = [LineSpec("A", 0.90)]
    model = MultiBandLineModel(lines, spectra, centers_mode="wavelength_solution")
    with pytest.raises(ValueError, match="band 2"):
        model.fit()


def test_duplicate_line_name_raises():
    spectra = _simple_spectra()
    lines = [LineSpec("A", 0.90), LineSpec("A", 1.30)]
    with pytest.raises(ValueError, match="duplicate line"):
        MultiBandLineModel(lines, spectra)


def test_duplicate_band_raises():
    lines = [LineSpec("A", 0.90)]
    spectra = _simple_spectra() + [_simple_spectra()[0]]
    with pytest.raises(ValueError, match="duplicate band"):
        MultiBandLineModel(lines, spectra)


def test_invalid_centers_mode_raises():
    lines = [LineSpec("A", 0.90)]
    spectra = _simple_spectra()
    with pytest.raises(ValueError, match="centers_mode"):
        MultiBandLineModel(lines, spectra, centers_mode="bogus")


def test_band_spectrum_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="shape mismatch"):
        BandSpectrum(1, np.array([1.0, 2.0]), np.array([1.0]), np.array([1.0, 1.0]))


def test_band_spectrum_rejects_non_monotonic_wavelength():
    with pytest.raises(ValueError, match="increasing"):
        BandSpectrum(1, np.array([1.0, 0.5]), np.array([1.0, 1.0]), np.array([1.0, 1.0]))


def test_line_table_classifies_by_combined_snr(noiseless_setup):
    lines, spectra = noiseless_setup
    model = MultiBandLineModel(lines, spectra)
    result = model.fit()
    table = result.line_table()
    assert set(table.columns) >= {"line", "center", "snr_comb", "classification",
                                  "chi2_consist", "upper_limit"}
    # Strong, noiseless lines should be classified as detected.
    assert (table.set_index("line").loc[["A", "B"], "classification"] == "detected").all()


def test_to_frame_is_serializable(noiseless_setup):
    lines, spectra = noiseless_setup
    model = MultiBandLineModel(lines, spectra)
    result = model.fit()
    frame = result.to_frame()
    assert "classification" in frame.columns
    # round-trips through a plain dict-of-arrays without error
    frame.to_dict(orient="list")

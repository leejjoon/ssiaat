"""Named-model API of Model / FitResults (positional .C/.contC stays)."""
import numpy as np
import pandas as pd
import pytest

from ssiaat.spherex_table import Model, FitResults

from conftest import PIXELS, TRUE_LINE_AMPS, TRUE_CONT_AMPS, TMPL_SHAPE


@pytest.fixture
def fitted(synthetic_stable, line_model, cont_model):
    model = Model({"line": line_model}, {"cont": cont_model})
    return model.least_square_fit(synthetic_stable, return_error=True)


def test_dict_models_named_lookup(fitted):
    for pix in PIXELS:
        np.testing.assert_allclose(fitted.coef("line")[pix],
                                   TRUE_LINE_AMPS[pix], rtol=1e-6)
        np.testing.assert_allclose(fitted.coef("cont")[pix],
                                   TRUE_CONT_AMPS[pix], rtol=1e-6)


def test_positional_access_unchanged(fitted):
    pd.testing.assert_series_equal(fitted.C[0], fitted.coef("line"))
    pd.testing.assert_series_equal(fitted.contC[0], fitted.coef("cont"))
    pd.testing.assert_series_equal(fitted.Cerr[0], fitted.err("line"))
    pd.testing.assert_series_equal(fitted.contCerr[0], fitted.err("cont"))


def test_list_input_auto_names(synthetic_stable, line_model, cont_model):
    model = Model([line_model], [cont_model])
    assert model.model_names == ["model0"]
    assert model.cont_model_names == ["cmodel0"]
    result = model.least_square_fit(synthetic_stable)
    assert result.coef("model0") is result.C[0]


def test_unknown_name_raises(fitted):
    with pytest.raises(KeyError, match="unknown model name"):
        fitted.coef("nope")


def test_err_requires_return_error(synthetic_stable, line_model, cont_model):
    model = Model({"line": line_model}, {"cont": cont_model})
    result = model.least_square_fit(synthetic_stable)  # no errors
    with pytest.raises(ValueError, match="return_error"):
        result.err("line")


def test_duplicate_names_rejected(line_model, cont_model):
    with pytest.raises(ValueError, match="unique"):
        Model({"a": line_model}, {"a": cont_model})


def test_to_frame(fitted):
    frame = fitted.to_frame()
    assert list(frame.columns) == ["line", "cont", "line_err", "cont_err"]
    assert list(frame.index) == sorted(PIXELS)
    np.testing.assert_allclose(frame["line"].values,
                               fitted.coef("line").values)


def test_image_by_name(fitted):
    image = fitted.image("line")
    assert image.shape == TMPL_SHAPE
    flat = np.ravel(np.asarray(image))
    assert np.isfinite(flat).sum() == len(PIXELS)


def test_model_required():
    with pytest.raises(TypeError):
        FitResults(np.array([0]), np.zeros((1, 1)))

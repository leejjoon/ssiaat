"""Linear model fitting on stables: Model and FitResults."""
import numpy as np
import pandas as pd

from itertools import chain
from .vectorized_lstsq import vectorized_lstsq_numpy, vectorized_lstsq_arrays

class FitResults:
    def __init__(self, idx, C, Cerr=None, *, model,
                 ssiaat_template_header=None):
        self.idx = idx
        self._C = C
        # Store coefficients as Series to leverage pandas index alignment
        self.C = [pd.Series(C[:, i], index=idx) for i in range(len(model.model_names))]
        self.contC = [pd.Series(C[:, i], index=idx) for i in range(len(model.model_names),
                                             len(model.all_model_names))]

        if ssiaat_template_header is not None:
            for s in chain(self.C, self.contC):
                s.attrs["ssiaat_template_header"] = ssiaat_template_header

        if Cerr is not None:
            self.Cerr = [pd.Series(Cerr[:, i], index=idx) for i in range(len(model.model_names))]
            self.contCerr = [pd.Series(Cerr[:, i], index=idx)
                             for i in range(len(model.model_names),
                                            len(model.all_model_names))]

            if ssiaat_template_header is not None:
                for s in chain(self.Cerr, self.contCerr):
                    s.attrs["ssiaat_template_header"] = ssiaat_template_header

        self._Cerr = Cerr
        self.model = model

    def _name_index(self, name):
        try:
            return self.model.all_model_names.index(name)
        except ValueError:
            raise KeyError(
                f"unknown model name {name!r};"
                f" available: {self.model.all_model_names}") from None

    def coef(self, name):
        """Coefficient Series (indexed by pixel) for the named model."""
        i = self._name_index(name)
        n = len(self.model.model_names)
        return self.C[i] if i < n else self.contC[i - n]

    def err(self, name):
        """Coefficient-error Series for the named model."""
        if self._Cerr is None:
            raise ValueError("no errors available: run least_square_fit"
                             " with return_error=True")
        i = self._name_index(name)
        n = len(self.model.model_names)
        return self.Cerr[i] if i < n else self.contCerr[i - n]

    def to_frame(self):
        """All coefficients (and errors, when fitted) as a DataFrame with
        named columns, indexed by pixel -- directly parquet-serializable."""
        data = {name: self.coef(name) for name in self.model.all_model_names}
        if self._Cerr is not None:
            for name in self.model.all_model_names:
                data[f"{name}_err"] = self.err(name)
        return pd.DataFrame(data)

    def image(self, name):
        """Coefficient map of the named model rendered on the template."""
        return self.coef(name).itable.to_image()

    def cont_sub(self, wvl, spec):
        # Using .reindex(wvl.index).values ensures we get a numpy array of the same length
        # as wvl, with values broadcasted to each duplicate index in the original order.
        cont = sum(amp.reindex(wvl.index).values * m(wvl)
                   for amp, m in zip(self.contC, self.model.cont_models))
        return spec - cont

    def norm(self, wvl, spec, param_i):
        # Using .reindex(wvl.index).values ensures we get a numpy array of the same length
        return spec / self.C[param_i].reindex(wvl.index).values

    def cont_sub_n_norm(self, wvl, spec, param_i):
        # Using .reindex(wvl.index).values ensures we get a numpy array of the same length
        spec_cont_sub = self.cont_sub(wvl, spec)
        return spec_cont_sub / self.C[param_i].reindex(wvl.index).values



class Model:
    """
    Linear combination of models (line models + continuum models).

    Each of `models` and `cont_models` is either a list of callables of
    wavelength (auto-named model0/model1/... and cmodel0/...) or a dict
    mapping names to callables, e.g.::

        Model({"br_a": get_br_a()}, {"cont": const()})

    Coefficients can then be read back by name: fitted.coef("br_a").
    """
    def __init__(self, models, cont_models):
        self.model_names, self.models = self._normalize(models, "model")
        self.cont_model_names, self.cont_models = self._normalize(cont_models,
                                                                  "cmodel")
        self.all_model_names = self.model_names + self.cont_model_names

        if len(set(self.all_model_names)) != len(self.all_model_names):
            raise ValueError("model names must be unique across models and"
                             f" cont_models: {self.all_model_names}")

    @staticmethod
    def _normalize(models, prefix):
        from collections.abc import Mapping
        if isinstance(models, Mapping):
            return list(models.keys()), list(models.values())
        models = list(models)
        return [f"{prefix}{i}" for i in range(len(models))], models

    def _populate_table_with_model_eval(self, stable, inplace=False):
        df = stable # stable is now a DataFrame
        k = {}
        for mid, m in chain(zip(self.model_names, self.models),
                            zip(self.cont_model_names, self.cont_models)):
            k[mid] = m(df["wvl"])

        if inplace:
            for n in k:
                df.loc[:, n] = k[n]

            df2 = df
        else:
            df2 = pd.DataFrame(k, index=df.index)
            # df2.loc[:, "tmpl_ind"] = df["tmpl_ind"]
            df2.loc[:, "wvl"] = df["wvl"]
            df2.loc[:, "image"] = df["image"]
            df2.loc[:, "variance"] = df["variance"]
            
        return df2

    def _least_square_fit(self, df, variance_column="variance", return_error=False):
        if return_error:
            C, C_err, idx = vectorized_lstsq_numpy(df, self.all_model_names, variance_column=variance_column, return_error=True)
            return idx, C, C_err
        else:
            C, idx = vectorized_lstsq_numpy(df, self.all_model_names,
                                            variance_column=variance_column,
                                            return_error=False)
            return idx, C

    def least_square_fit(self, stable, variance_column="variance", return_error=False):
        # Evaluate models straight into arrays and fit in one pure-numpy
        # pass -- no full-length DataFrame copy of wvl/image/variance.
        wvl = stable["wvl"].to_numpy()
        model_arrays = [m(wvl) for m in chain(self.models, self.cont_models)]
        target = stable["image"].to_numpy()
        weights = (1.0 / stable[variance_column].to_numpy(dtype="float64")
                   if variance_column is not None else None)

        codes, uniques = pd.factorize(stable.index, sort=True)
        idx = np.asarray(uniques)

        result = vectorized_lstsq_arrays(model_arrays, target, codes,
                                         len(idx), weights=weights,
                                         return_error=return_error)
        if return_error:
            idx_C_Cerr = (idx, *result)
        else:
            idx_C_Cerr = (idx, result)

        ssiaat_template_header = stable.attrs.get("ssiaat_template_header", None)
        return FitResults(*idx_C_Cerr, model=self,
                          ssiaat_template_header=ssiaat_template_header)

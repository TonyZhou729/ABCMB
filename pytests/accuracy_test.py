import numpy as np
import pytest
from classy import Class

from abcmb import species
from abcmb.main import Model

# JAX platform, x64, and debug_nans are configured in conftest.py.
np.seterr(all="raise")


def test_accuracy_checker(h=0.6762):
    ellmin = 2
    ellmax = 2500
    try:
        # Setup

        params = {
            "h": h,
            "omega_cdm": 0.1193,
            "omega_b": 0.0225,
            "A_s": 2.12424e-9,
            "n_s": 0.9709,
            "Neff": 3.044,
            "YHe": 0.245,
            "TCMB0": 2.34865418e-4,
            "N_nu_massive": 0,
            "T_nu_massive": 0.71611,
            "m_nu_massive": 0.06,
            "tau_reion": 0.0544,
            "Delta_z_reion": 0.5,
            "z_reion_He": 3.5,
            "Delta_z_reion_He": 0.5,
            "exp_reion": 1.5,
        }

        if params["N_nu_massive"] > 0:
            user_species = (species.MassiveNeutrino,)
        else:
            user_species = None

        model = Model(
            user_species=user_species,
            output_Cl=True,
            l_max=ellmax,
            lensing=True,
            output_Pk=True,
            output_k_max=0.5,
            l_max_g=12,
            l_max_pol_g=10,
            l_max_ur=17,
            l_max_ncdm=17,
        )
        full_params = model.add_derived_parameters(params)

        # CLASS
        CLASS_params = {
            "output": "mPk, tCl, pCl, lCl"
            if model.specs["lensing"]
            else "mPk, tCl, pCl",
            # "temperature_contributions" : "tsw",
            "l_max_scalars": ellmax,
            "P_k_max_1/Mpc": model.specs["output_k_max"],
            "lensing": "yes" if model.specs["lensing"] else "no",
            "accurate_lensing": 1,
            "H0": full_params["h"] * 100,
            "omega_b": full_params["omega_b"],
            "omega_cdm": full_params["omega_cdm"],
            "A_s": full_params["A_s"],
            "n_s": full_params["n_s"],
            "N_ur": full_params["Neff"],
            "YHe": full_params["YHe"],
            "N_ncdm": full_params["N_nu_massive"],
            # "reio_parametrization" : "reio_none",
            "reio_parametrization": "reio_camb",
            "tau_reio": params["tau_reion"],
            "reionization_width": params["Delta_z_reion"],
            "helium_fullreio_redshift": params["z_reion_He"],
            "helium_fullreio_width": params["Delta_z_reion_He"],
            "reionization_exponent": params["exp_reion"],
            "l_max_g": model.specs["l_max_g"],
            "l_max_pol_g": model.specs["l_max_pol_g"],
            "l_max_ur": model.specs["l_max_ur"],
            "l_max_ncdm": model.specs["l_max_ncdm"],
        }

        CLASS_Model = Class()
        CLASS_Model.set(CLASS_params)
        if full_params["N_nu_massive"] > 0:
            CLASS_Model.set(
                {
                    "m_ncdm": full_params["m_nu_massive"],
                    "T_ncdm": full_params["T_nu_massive"],
                }
            )

        CLASS_Model.compute()
        if model.specs["lensing"]:
            cl = CLASS_Model.lensed_cl(ellmax)
        else:
            cl = CLASS_Model.raw_cl(ellmax)
        cltt = cl["tt"][ellmin:]
        clee = cl["ee"][ellmin:]

        # ABCMB

        output = model(params)

        ABC_tt = output.ClTT
        ABC_ee = output.ClEE

        # Compare Cltt
        err_tt = abs(cltt - ABC_tt) / cltt
        print(err_tt.max())

        # Compare Clee
        err_ee = abs(clee - ABC_ee) / clee
        print(err_ee.max())

        # Compare P(k)
        ABC_Pk = output.Pk
        ABC_k = output.k
        CLA_Pk = np.vectorize(CLASS_Model.pk)(ABC_k, 0.0)
        err_Pk = abs(CLA_Pk - ABC_Pk) / CLA_Pk
        print(err_Pk.max())

        assert max(err_tt) <= 0.01, f"Accuracy check failed at TT: {err_tt}"
        assert max(err_ee) <= 0.01, f"Accuracy check failed at EE: {err_ee}"
        assert max(err_Pk) <= 0.01, f"Accuracy check failed at P(k): {err_Pk}"

    except Exception as e:
        pytest.fail(f"accuracy_checks raised an exception: {e}")


# print(test_accuracy_checker())

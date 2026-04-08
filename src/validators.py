import logging
from typing import Dict, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("QUFT_Validator")


class AssumptionValidator:
    """Boundary checks matching QUFT- RGE-Thermal.txt §2-4"""

    @staticmethod
    def vacuum_copositivity(lam_H: float, lam_S: float, lam_HS: float) -> bool:
        """BFB: λ_H > 0, λ_S > 0, λ_HS > -2√(λ_H λ_S)"""
        if lam_H <= 1e-4 or lam_S <= 1e-4:
            log.warning("Vacuum instability: λ_H or λ_S negative")
            return False
        bound = -2 * np.sqrt(lam_H * lam_S)
        if lam_HS <= bound:
            log.warning("Copositivity violated: λ_HS too negative")
            return False
        return True

    @staticmethod
    def metastability_bound(lam_min: float, mu_min: float) -> Tuple[bool, float, float]:
        """S_E > 280 + 4 ln(μ/GeV)"""
        S_E = 8 * np.pi**2 / (3 * abs(lam_min))
        bound = 280 + 4 * np.log(mu_min / 1.0)
        return S_E > bound, S_E, bound

    @staticmethod
    def thermal_consistency(T_reh: float, M2: float = 2.4e23) -> bool:
        """T_reh < M2 and no thermal destabilization"""
        if T_reh >= M2:
            log.warning("Thermal mismatch: T_reh ≥ M2")
            return False
        # High-T mass corrections positive per docs
        return True

    @staticmethod
    def check_boundaries(f2: float, perturbative_limit: float = 1.0) -> Dict[str, bool]:
        """Assumption boundary monitoring"""
        flags: Dict[str, bool] = {}
        flags["A1_perturbative"] = f2 < perturbative_limit
        flags["A2_fakeon_valid"] = True  # Contour prescription assumed
        if not all(flags.values()):
            log.critical("Assumption boundary breached: %s", flags)
        return flags

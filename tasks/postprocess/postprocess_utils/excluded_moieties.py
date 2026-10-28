"""
List of moieties that are excluded from the analysis.
"""

EXCLUDED_MOIETIES = [
    # "secondary_carbon",
    # "tertiary_carbon",
    # "quaternary_carbon",
    # "ether",
    # "trivalent_nitrogen",
    # "arsenide_ion",
    # "boride_ion",
    # "lithium_cation",
    # "oxygen_anion",
    # "phosphorus_anion",
    # "sulfur_anion",
    "ring_5"
]

EXCLUDED_VISUALIZATION_MOLECULES = [
    "[Li+].F[P-](F)(F)(F)(F)F",
    "[Li+].F[B-](F)(F)(F)",
    "[Li+].C(F)(F)(F)S(=O)(=O)[N-]S(=O)(=O)C(F)(F)F",
    "[Li+].F[B-](F)(F)OC(C(F)(F)(F))(C(F)(F)(F))C(F)(F)(F)",
    "[Li+].[N-](S(=O)(=O)F)S(=O)(=O)F",
    "[Li+].F[As-](F)(F)(F)(F)F",
    "[Li+].C(=O)(C(F)(F)F)[O-]",
    "[Li+].C(C(F)(F)S(=O)(=O)[N-]S(=O)(=O)C(C(F)(F)F)(F)F)(F)(F)F",
    "[Li+].C(F)(F)(F)S(=O)(=O)[N-]S(=O)(=O)C",
    "[Li+].C(F)(F)(F)S(=O)(=O)[N-]S(=O)(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F",
    "[Li+].FP(F)(=O)([O-])",
    "[Li+].[B-]1(OC(=O)C(=O)O1)(F)F",
    "[Li+].[B-]12(OC(=O)C(=O)O1)OC(=O)C(=O)O2",
    "[Li+].[N+](=O)([O-])[O-]",
    "[Li+].[O-]Cl(=O)(=O)=O",
    "[Li+].[O-]P(=O)(F)F",
    "[Li+].[P-](F)(F)12(OC(=O)C(=O)O1)OC(=O)C(=O)O2",
    "[Li+][S-]SSS[S-][Li+]",
    "COCCOC",
]

# Moieties excluded from visualization (ionic species)
EXCLUDED_VISUALIZATION_MOIETIES = [
    "ammonium_ion",
    "arsenide_ion",
    "boride_ion",
    "lithium_cation",
    "nitrogen_anion",
    "oxygen_anion",
    "phosphorus_anion",
    "sulfur_anion",
]

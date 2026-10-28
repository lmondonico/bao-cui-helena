MOIETIES = {
    # Note that groups are indexed starting from 2
    # 0 index is reserved for padding token
    # Carbon-based groups
    "methyl": "[CX4H3]",
    "secondary_carbon": "[#6X4H2]",
    "tertiary_carbon": "[#6X4H1]",
    "quaternary_carbon": "[#6X4H0]",
    "alkene": "[CX3]=[CX3]",
    "benzene": "c1ccccc1",
    "naphthalene": "c12ccccc1cccc2",
    # Oxygen-containing groups
    "ether": "[OX2]",
    "carbonyl": "[C](=O)",
    "carboxylate": "[CX3](=O)[O-]",
    "carbonate_ester": "[#6][OX2]C(=O)[OX2][#6]",
    # Nitrogen-containing groups
    "trivalent_nitrogen": "[NX3]",
    "amide": "[NX3][CX3](=[OX1])[#6]",
    "nitrile": "[NX1]#[CX2]",
    # Sulfur-containing groups
    "disulfide": "[SX2;!$([S-])][SX2;!$([S-])]",
    "trisulfide": "[SX2;!$([S-])][SX2;!$([S-])][SX2;!$([S-])]",
    "sulfinyl": "[SX3](=O)",
    "sulfonyl": "[SX4](=O)(=O)",
    # Phosphorus-containing groups
    "phosphoryl": "[P](=O)",
    "cyclotriphosphazene": "[N]1=P[N]=P[N]=P1",
    # Boron-containing groups
    "trigonal_borate": "[BX3](O)(O)O",
    # Silicon-containing groups
    "silyl": "[SiX4]",
    # Halogens
    "fluoro": "[FX1]",
    "chloro": "[ClX1]",
    "perchlorate_chlorine": "[ClX4]",
    "bromo": "[BrX1]",
    "iodo": "[IX1]",
    # Ionic groups
    "ammonium_ion": "[NX4+]",
    "arsenide_ion": "[As-]",
    "boride_ion": "[B-]",
    "lithium_cation": "[Li+]",
    "nitrogen_anion": "[N-]",
    "oxygen_anion": "[O-]",
    "phosphorus_anion": "[P-]",
    "sulfur_anion": "[S-]",
    "nitrate_ion": "[N+](=O)([O-])[O-]",
}

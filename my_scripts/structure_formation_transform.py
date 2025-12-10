from bidict import bidict
element_list = [
    # 0
    'X',
    # 1
    'H', 'He',
    # 2
    'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    # 3
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    # 4
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    # 5
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    # 6
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
    'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
    'Po', 'At', 'Rn',
    # 7
    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk',
    'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc',
    'Lv', 'Ts', 'Og']
ELEMENT_MAP = bidict({value: index for index, value in enumerate(element_list)})
WYCKOFF_MAP = bidict(
    {'X': 0} |
    {chr(ord('a') + i): i + 1 for i in range(26)} |
    {'A': 27}
)
def convert_element_symbol_number(value):
    """Convert between element symbol and index."""

    if isinstance(value, str):
        # symbol -> index
        return ELEMENT_MAP[value]

    if isinstance(value, int):
        # index -> symbol
        return ELEMENT_MAP.inverse[value]


    raise TypeError(f"Value must be int or str, got {type(value)}")

def convert_wyckoff_symbol_number(value):
    """Convert between wyckoff symbol and index."""
    if isinstance(value, str):
        return WYCKOFF_MAP[value]

    if isinstance(value, int):
        return WYCKOFF_MAP.inverse[value]

    raise TypeError(f"Value must be int or str, got {type(value)}")

if __name__ =="__main__":
    print(convert_element_symbol_number("Hg"))
    print(convert_element_symbol_number(1))
    print(convert_wyckoff_symbol_number("a"))
    print(convert_wyckoff_symbol_number(1))
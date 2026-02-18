#!/usr/bin/env python3
"""
Diagnostic: For fragments excluded by the accuracy filter (wrong c/z match),
test whether they match b/y ions at the same cleavage position.

If observed m/z matches b or y theoretical within ppm, the peak may be a
CID/HCD fragment mis-assigned as ETD/ECD (c/z).

b/y formulas aligned with Comet: b = dNtermProton + sum; y = dCtermOH2Proton + sum;
m/z = (mass + (z-1)*PROTON) / z.

Usage:
    python test_excluded_fragments_as_by.py
    python test_excluded_fragments_as_by.py --input data/comet_frags_perc_openMS_confidence.csv --ppm 20
"""

import argparse
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(_SCRIPT_DIR, 'data', 'comet_frags_perc_openMS_confidence.csv')
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from visualization.chromatograms import (
    get_best_theoretical_match,
    PROTON_MASS,
    _AA_MASS,
    _Y_ION_OFFSET,
)


def get_b_y_theoretical_at_cleavage(peptide, ion_name, charge):
    """
    For Comet ion (c5/z6/z1_7) at cleavage k, return (b_mz, y_mz) at same cleavage.
    c5 = N-term 1..5, cleavage after 5. b5 = same. y_{L-5} = C-term 6..L. Same cleavage.
    z5 = C-term first 5 from C-end; cleavage k = L-5. b_{L-5} and y_5 share that cleavage.
    """
    seq = ''.join(c for c in str(peptide).upper() if c in _AA_MASS)
    if not seq:
        return None, None
    n = len(seq)
    k = None
    if ion_name.startswith('c') and ion_name[1:].isdigit():
        k = int(ion_name[1:])
    elif ion_name.startswith('z1_'):
        k = n - int(ion_name.split('_')[-1])
    elif ion_name.startswith('z') and ion_name[1:].isdigit():
        k = n - int(ion_name[1:])
    if k is None or k < 1 or k >= n:
        return None, None
    # b_k: N-term residues 1..k, mass = sum + H ([MH]+)
    mass_b = PROTON_MASS
    for i in range(k):
        mass_b += _AA_MASS.get(seq[i], 0)
    # y_{n-k}: C-term residues k..n-1 ([MH]+ = H2O + PROTON + sum, matches Comet dCtermOH2Proton)
    mass_y = _Y_ION_OFFSET
    for i in range(k, n):
        mass_y += _AA_MASS.get(seq[i], 0)
    # m/z = (MH + (z-1)*PROTON) / z
    protons = (charge - 1) * PROTON_MASS
    b_mz = (mass_b + protons) / charge
    y_mz = (mass_y + protons) / charge
    return b_mz, y_mz


def main():
    ap = argparse.ArgumentParser(description='Test if excluded fragments match b/y ions')
    ap.add_argument('--input', '-i', default=DEFAULT_CSV)
    ap.add_argument('--ppm', type=float, default=5.0)
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found")
        sys.exit(1)

    import pandas as pd
    df = pd.read_csv(args.input, sep=',', skiprows=1 if 'CometVersion' in open(args.input).readline() else 0,
                     engine='python', quotechar='"', on_bad_lines='warn')

    col_ions = 'matched fragment ions'
    col_mz = 'matched fragment ion mz'
    if col_ions not in df.columns or col_mz not in df.columns:
        print("Error: CSV must have matched fragment ions and mz columns")
        sys.exit(1)

    results = []  # (ion_name, obs_mz, best_alt, alt_type, ppm)
    n_excluded = 0

    for idx in df.index:
        row = df.loc[idx]
        peptide = row.get('plain_peptide') or row.get('sequence') or ''
        if not peptide or not isinstance(peptide, str):
            continue
        ions = [p.strip() for p in str(row.get(col_ions, '')).split(',') if p.strip()]
        mzs = [p.strip() for p in str(row.get(col_mz, '')).split(',') if p.strip()]
        if len(ions) != len(mzs):
            continue

        for name, mz_s in zip(ions, mzs):
            if not name or not mz_s:
                continue
            if not ((name.startswith('c') and name[1:].isdigit()) or
                    (name.startswith('z1_') and name.split('_')[-1].isdigit()) or
                    (name.startswith('z') and name[1:].isdigit())):
                continue
            try:
                obs_mz = float(mz_s)
            except ValueError:
                continue
            if obs_mz <= 0:
                continue
            match = get_best_theoretical_match(name, peptide, obs_mz)
            if match is None:
                continue
            theo_mz, best_charge, ppm_cz = match
            passed_cz = ppm_cz <= args.ppm
            if passed_cz:
                continue  # not excluded

            n_excluded += 1

            # Excluded: test b/y at same cleavage, all charges
            best_alt = None
            best_alt_type = None
            best_alt_ppm = float('inf')
            for z in (1, 2, 3, 4, 5):
                b_mz, y_mz = get_b_y_theoretical_at_cleavage(peptide, name, z)
                for alt_mz, alt_type in [(b_mz, 'b'), (y_mz, 'y')]:
                    if alt_mz is None or alt_mz <= 0:
                        continue
                    ppm = abs(obs_mz - alt_mz) / alt_mz * 1e6
                    if ppm < best_alt_ppm:
                        best_alt_ppm = ppm
                        best_alt = alt_mz
                        best_alt_type = alt_type

            if best_alt is not None and best_alt_ppm <= args.ppm:
                results.append((name, obs_mz, best_alt, best_alt_type, best_alt_ppm))

            n_excluded += 1
    n_match_by = len(results)
    print(f"Excluded fragments (failed c/z ppm): {n_excluded}")
    print(f"Match b/y within {args.ppm} ppm: {n_match_by} ({100*n_match_by/n_excluded:.1f}%)" if n_excluded else "Match b/y: 0")

    if results:
        by_type = {}
        for r in results:
            t = r[3]
            by_type[t] = by_type.get(t, 0) + 1
        print(f"  -> b: {by_type.get('b', 0)}, y: {by_type.get('y', 0)}")
        print("\nSample matches (first 10):")
        for name, obs, theo, alt_type, ppm in results[:10]:
            print(f"  {name} obs={obs:.3f} -> {alt_type} theo={theo:.3f} ppm={ppm:.1f}")

    return 0


if __name__ == '__main__':
    sys.exit(main())

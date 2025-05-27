#!/usr/bin/env python3
import cv2
import numpy as np
import argparse
from skimage.feature import local_binary_pattern

def chi2_histogram(img1, img2, bins=(8,8,8), eps=1e-10):
    """Distance de Chi² entre histogrammes BGR 3D."""
    hist1 = cv2.calcHist([img1], [0,1,2], None, bins, [0,256]*3).flatten()
    hist2 = cv2.calcHist([img2], [0,1,2], None, bins, [0,256]*3).flatten()
    hist1 /= (hist1.sum() + eps); hist2 /= (hist2.sum() + eps)
    return 0.5 * np.sum((hist1 - hist2)**2 / (hist1 + hist2 + eps))

def lbp_distance(img1, img2, P=8, R=1, bins=24):
    """
    Distance L2 entre histogrammes de LBP sur la luminance.
    P = nombre de voisins, R = rayon.
    """
    # 1) Extraire luminance et normaliser
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 2) Calculer LBP
    lbp1 = local_binary_pattern(gray1, P, R, method='uniform')
    lbp2 = local_binary_pattern(gray2, P, R, method='uniform')

    # 3) Histogrammes normalisés
    hist1, _ = np.histogram(lbp1.ravel(), bins=bins, range=(0, bins), density=True)
    hist2, _ = np.histogram(lbp2.ravel(), bins=bins, range=(0, bins), density=True)

    # 4) Distance L2
    return float(np.linalg.norm(hist1 - hist2))

def main():
    parser = argparse.ArgumentParser(
        description="Évaluation colorimétrique + texture (LBP) entre deux images")
    parser.add_argument('--ref',   required=True, help="Image de référence")
    parser.add_argument('--synth', required=True, help="Image synthétisée")
    parser.add_argument('--bins',  nargs=3, type=int, default=[8,8,8],
                        help="Buckets par canal BGR pour χ² histo")
    parser.add_argument('--P',     type=int, default=8,
                        help="Nombre de voisins pour LBP")
    parser.add_argument('--R',     type=float, default=1.0,
                        help="Rayon pour LBP")
    parser.add_argument('--lbp_bins', type=int, default=24,
                        help="Buckets pour histogramme LBP")
    args = parser.parse_args()

    # Chargement
    ref   = cv2.imread(args.ref,   cv2.IMREAD_COLOR)
    synth = cv2.imread(args.synth, cv2.IMREAD_COLOR)
    if ref is None or synth is None:
        parser.error("Impossible de lire l’une des images (--ref ou --synth invalide).")

    # Calcul des métriques
    chi2 = chi2_histogram(ref, synth, bins=tuple(args.bins))
    lbp  = lbp_distance(ref, synth, P=args.P, R=args.R, bins=args.lbp_bins)

    # Affichage
    print(f"Chi² histogram distance (bins={args.bins}): {chi2:.4f}")
    print(f"LBP histogram L2 distance (P={args.P}, R={args.R}, bins={args.lbp_bins}): {lbp:.4f}")

    # Score combiné (optionnel)
    combined = chi2 + lbp
    print(f"Combined score (χ² + L2 LBP): {combined:.4f}")

if __name__ == '__main__':
    main()

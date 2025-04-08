import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
import random
import time
from skimage.metrics import structural_similarity as ssim

class ImageQuilting:
    def __init__(self, block_size=32, overlap=6, tolerance=0.1):
        """
        Initialisation de l'algorithme Image Quilting
        
        Args:
            block_size (int): Taille des blocs (carrés) en pixels
            overlap (int): Taille du chevauchement entre les blocs adjacents
            tolerance (float): Tolérance d'erreur lors de la sélection des blocs (0.1 = 10%)
        """
        self.block_size = block_size
        self.overlap = overlap
        self.tolerance = tolerance
    
    def synthesize_texture(self, input_texture, output_height, output_width):
        """
        Synthèse d'une texture plus grande à partir d'une texture d'entrée
        
        Args:
            input_texture (np.array): Image de texture d'entrée
            output_height (int): Hauteur de l'image de sortie
            output_width (int): Largeur de l'image de sortie
            
        Returns:
            np.array: Texture synthétisée
        """
        # Vérifier si l'image est en niveaux de gris ou en couleur
        is_color = len(input_texture.shape) == 3
        
        # Initialiser l'image de sortie
        if is_color:
            output = np.zeros((output_height, output_width, 3), dtype=np.uint8)
        else:
            output = np.zeros((output_height, output_width), dtype=np.uint8)
        
        # Calculer le nombre de blocs nécessaires
        n_blocks_h = int(np.ceil((output_height - self.overlap) / (self.block_size - self.overlap)))
        n_blocks_w = int(np.ceil((output_width - self.overlap) / (self.block_size - self.overlap)))
        
        # Pour chaque position de bloc
        for i in range(n_blocks_h):
            for j in range(n_blocks_w):
                # Calculer la position du coin supérieur gauche du bloc
                y = i * (self.block_size - self.overlap)
                x = j * (self.block_size - self.overlap)
                
                # Taille du bloc actuel (peut être plus petit aux bords)
                block_h = min(self.block_size, output_height - y)
                block_w = min(self.block_size, output_width - x)
                
                # Déterminer les contraintes de chevauchement
                constraints = {}
                if i > 0:  # Il y a un bloc au-dessus
                    constraints['top'] = output[y:y+self.overlap, x:x+block_w]
                if j > 0:  # Il y a un bloc à gauche
                    constraints['left'] = output[y:y+block_h, x:x+self.overlap]
                
                # Sélectionner un bloc approprié
                block = self._find_matching_block(input_texture, block_h, block_w, constraints)
                
                # Appliquer la coupe minimale d'erreur si nécessaire
                if i > 0 and j > 0:  # Chevauchement avec le bloc de gauche et celui du haut
                    block = self._minimum_error_boundary_cut_both(
                        output[y:y+self.overlap, x:x+block_w],  # Chevauchement haut
                        output[y:y+block_h, x:x+self.overlap],  # Chevauchement gauche
                        block, 
                        self.overlap
                    )
                elif i > 0:  # Seulement chevauchement avec le bloc du haut
                    block = self._minimum_error_boundary_cut_horizontal(
                        output[y:y+self.overlap, x:x+block_w], 
                        block, 
                        self.overlap
                    )
                elif j > 0:  # Seulement chevauchement avec le bloc de gauche
                    block = self._minimum_error_boundary_cut_vertical(
                        output[y:y+block_h, x:x+self.overlap], 
                        block, 
                        self.overlap
                    )
                
                # Placer le bloc dans l'image de sortie
                output[y:y+block_h, x:x+block_w] = block
        
        return output
    
    def _find_matching_block(self, input_texture, block_h, block_w, constraints):
        """
        Trouver un bloc dans la texture d'entrée qui correspond aux contraintes
        
        Args:
            input_texture (np.array): Texture d'entrée
            block_h (int): Hauteur du bloc à extraire
            block_w (int): Largeur du bloc à extraire
            constraints (dict): Contraintes de chevauchement
            
        Returns:
            np.array: Bloc sélectionné
        """
        # Dimensions de la texture d'entrée
        input_h, input_w = input_texture.shape[:2]
        
        # Générer toutes les positions possibles pour le bloc
        valid_y = range(0, input_h - block_h + 1)
        valid_x = range(0, input_w - block_w + 1)
        
        # Si pas de contraintes, choisir un bloc aléatoire
        if not constraints:
            y = random.choice(valid_y)
            x = random.choice(valid_x)
            return input_texture[y:y+block_h, x:x+block_w].copy()
        
        # Calculer l'erreur pour chaque position possible
        min_error = float('inf')
        best_blocks = []
        
        for y in valid_y:
            for x in valid_x:
                error = 0
                
                # Vérifier la contrainte du haut
                if 'top' in constraints:
                    top_constraint = constraints['top']
                    top_overlap = input_texture[y:y+self.overlap, x:x+block_w]
                    error += np.sum((top_overlap - top_constraint) ** 2)
                
                # Vérifier la contrainte de gauche
                if 'left' in constraints:
                    left_constraint = constraints['left']
                    left_overlap = input_texture[y:y+block_h, x:x+self.overlap]
                    error += np.sum((left_overlap - left_constraint) ** 2)
                
                # Normaliser l'erreur
                if 'top' in constraints and 'left' in constraints:
                    error /= (self.overlap * block_w + self.overlap * block_h - self.overlap**2)
                elif 'top' in constraints:
                    error /= (self.overlap * block_w)
                elif 'left' in constraints:
                    error /= (self.overlap * block_h)
                
                # Mettre à jour le meilleur bloc
                if error < min_error:
                    min_error = error
                    best_blocks = [(y, x)]
                elif error < min_error * (1 + self.tolerance):
                    best_blocks.append((y, x))
        
        # Choisir aléatoirement parmi les meilleurs blocs
        y, x = random.choice(best_blocks)
        return input_texture[y:y+block_h, x:x+block_w].copy()
    
    def _minimum_error_boundary_cut_vertical(self, left_block, new_block, overlap):
        """
        Calcule la coupe d'erreur minimale verticale entre deux blocs
        
        Args:
            left_block (np.array): Bloc de gauche (déjà placé)
            new_block (np.array): Nouveau bloc à placer
            overlap (int): Taille du chevauchement
            
        Returns:
            np.array: Nouveau bloc avec la coupe appliquée
        """
        block_h, _ = new_block.shape[:2]
        
        # Calculer la surface d'erreur
        error_surface = (left_block - new_block[:, :overlap]) ** 2
        
        # Si l'image est en couleur, sommer les erreurs sur les canaux
        if len(error_surface.shape) == 3:
            error_surface = np.sum(error_surface, axis=2)
        
        # Initialiser le tableau d'erreurs cumulées
        cumulative_error = np.zeros_like(error_surface)
        cumulative_error[0, :] = error_surface[0, :]
        
        # Remplir le tableau d'erreurs cumulées
        for i in range(1, block_h):
            for j in range(overlap):
                if j == 0:
                    cumulative_error[i, j] = error_surface[i, j] + min(
                        cumulative_error[i-1, j],
                        cumulative_error[i-1, j+1]
                    )
                elif j == overlap - 1:
                    cumulative_error[i, j] = error_surface[i, j] + min(
                        cumulative_error[i-1, j-1],
                        cumulative_error[i-1, j]
                    )
                else:
                    cumulative_error[i, j] = error_surface[i, j] + min(
                        cumulative_error[i-1, j-1],
                        cumulative_error[i-1, j],
                        cumulative_error[i-1, j+1]
                    )
        
        # Trouver le chemin de coût minimal
        path = np.zeros(block_h, dtype=np.int32)
        path[-1] = np.argmin(cumulative_error[-1, :])
        
        for i in range(block_h - 2, -1, -1):
            j = path[i+1]
            if j == 0:
                path[i] = np.argmin([cumulative_error[i, j], cumulative_error[i, j+1]])
            elif j == overlap - 1:
                path[i] = j - 1 + np.argmin([cumulative_error[i, j-1], cumulative_error[i, j]])
            else:
                path[i] = j - 1 + np.argmin([cumulative_error[i, j-1], cumulative_error[i, j], cumulative_error[i, j+1]])
        
        # Créer un masque pour la coupe
        mask = np.ones_like(new_block, dtype=bool)
        for i in range(block_h):
            mask[i, :path[i]+1] = False
        
        # Appliquer le masque
        result = new_block.copy()
        if len(new_block.shape) == 3:
            for c in range(3):
                result[:, :overlap, c] = np.where(mask[:, :overlap, 0], new_block[:, :overlap, c], left_block[:, :, c])
        else:
            result[:, :overlap] = np.where(mask[:, :overlap], new_block[:, :overlap], left_block)
        
        return result
    
    def _minimum_error_boundary_cut_horizontal(self, top_block, new_block, overlap):
        """
        Calcule la coupe d'erreur minimale horizontale entre deux blocs
        
        Args:
            top_block (np.array): Bloc du haut (déjà placé)
            new_block (np.array): Nouveau bloc à placer
            overlap (int): Taille du chevauchement
            
        Returns:
            np.array: Nouveau bloc avec la coupe appliquée
        """
        _, block_w = new_block.shape[:2]
        
        # Calculer la surface d'erreur
        error_surface = (top_block - new_block[:overlap, :]) ** 2
        
        # Si l'image est en couleur, sommer les erreurs sur les canaux
        if len(error_surface.shape) == 3:
            error_surface = np.sum(error_surface, axis=2)
        
        # Initialiser le tableau d'erreurs cumulées
        cumulative_error = np.zeros_like(error_surface)
        cumulative_error[:, 0] = error_surface[:, 0]
        
        # Remplir le tableau d'erreurs cumulées
        for j in range(1, block_w):
            for i in range(overlap):
                if i == 0:
                    cumulative_error[i, j] = error_surface[i, j] + min(
                        cumulative_error[i, j-1],
                        cumulative_error[i+1, j-1]
                    )
                elif i == overlap - 1:
                    cumulative_error[i, j] = error_surface[i, j] + min(
                        cumulative_error[i-1, j-1],
                        cumulative_error[i, j-1]
                    )
                else:
                    cumulative_error[i, j] = error_surface[i, j] + min(
                        cumulative_error[i-1, j-1],
                        cumulative_error[i, j-1],
                        cumulative_error[i+1, j-1]
                    )
        
        # Trouver le chemin de coût minimal
        path = np.zeros(block_w, dtype=np.int32)
        path[-1] = np.argmin(cumulative_error[:, -1])
        
        for j in range(block_w - 2, -1, -1):
            i = path[j+1]
            if i == 0:
                path[j] = np.argmin([cumulative_error[i, j], cumulative_error[i+1, j]])
            elif i == overlap - 1:
                path[j] = i - 1 + np.argmin([cumulative_error[i-1, j], cumulative_error[i, j]])
            else:
                path[j] = i - 1 + np.argmin([cumulative_error[i-1, j], cumulative_error[i, j], cumulative_error[i+1, j]])
        
        # Créer un masque pour la coupe
        mask = np.ones_like(new_block, dtype=bool)
        for j in range(block_w):
            mask[:path[j]+1, j] = False
        
        # Appliquer le masque
        result = new_block.copy()
        if len(new_block.shape) == 3:
            for c in range(3):
                result[:overlap, :, c] = np.where(mask[:overlap, :, 0], new_block[:overlap, :, c], top_block[:, :, c])
        else:
            result[:overlap, :] = np.where(mask[:overlap, :], new_block[:overlap, :], top_block)
        
        return result
    
    def _minimum_error_boundary_cut_both(self, top_block, left_block, new_block, overlap):
        """
        Calcule la coupe d'erreur minimale pour les deux directions (verticale et horizontale)
        
        Args:
            top_block (np.array): Bloc du haut (déjà placé)
            left_block (np.array): Bloc de gauche (déjà placé)
            new_block (np.array): Nouveau bloc à placer
            overlap (int): Taille du chevauchement
            
        Returns:
            np.array: Nouveau bloc avec les coupes appliquées
        """
        # Résoudre le problème de chevauchement en L
        block_h, block_w = new_block.shape[:2]
        
        # 1. Appliquer d'abord la coupe horizontale (avec le bloc du haut)
        result_horizontal = self._minimum_error_boundary_cut_horizontal(top_block, new_block, overlap)
        
        # 2. Appliquer ensuite la coupe verticale (avec le bloc de gauche)
        result_vertical = self._minimum_error_boundary_cut_vertical(left_block, new_block, overlap)
        
        # 3. Fusionner les deux résultats
        result = new_block.copy()
        
        # Appliquer les coupes, tout en prenant soin du coin de chevauchement (overlap x overlap)
        if len(new_block.shape) == 3:
            # Pour les images en couleur
            for c in range(3):
                # Appliquer le résultat horizontal pour la bande supérieure
                result[:overlap, overlap:, c] = result_horizontal[:overlap, overlap:, c]
                
                # Appliquer le résultat vertical pour la bande gauche
                result[overlap:, :overlap, c] = result_vertical[overlap:, :overlap, c]
                
                # Pour le coin supérieur gauche, faire une moyenne pondérée
                # ou utiliser un autre critère pour fusionner les deux résultats
                for i in range(overlap):
                    for j in range(overlap):
                        # Pondération diagonale: plus proche du coin, plus d'influence
                        alpha = (i + j) / (2 * overlap)
                        result[i, j, c] = alpha * result_horizontal[i, j, c] + (1 - alpha) * result_vertical[i, j, c]
        else:
            # Pour les images en niveaux de gris
            # Appliquer le résultat horizontal pour la bande supérieure
            result[:overlap, overlap:] = result_horizontal[:overlap, overlap:]
            
            # Appliquer le résultat vertical pour la bande gauche
            result[overlap:, :overlap] = result_vertical[overlap:, :overlap]
            
            # Pour le coin supérieur gauche, faire une moyenne pondérée
            for i in range(overlap):
                for j in range(overlap):
                    alpha = (i + j) / (2 * overlap)
                    result[i, j] = alpha * result_horizontal[i, j] + (1 - alpha) * result_vertical[i, j]
        
        return result
    
    def texture_transfer(self, source_texture, target_image, num_iterations=3, alpha=0.8):
        """
        Transfert de texture d'une image source à une image cible
        
        Args:
            source_texture (np.array): Texture source
            target_image (np.array): Image cible
            num_iterations (int): Nombre d'itérations pour le raffinement
            alpha (float): Poids pour l'équilibre entre texture et correspondance
            
        Returns:
            np.array: Image avec la texture transférée
        """
        # Convertir en niveaux de gris pour la correspondance
        if len(source_texture.shape) == 3:
            source_gray = cv2.cvtColor(source_texture, cv2.COLOR_RGB2GRAY)
        else:
            source_gray = source_texture.copy()
            
        if len(target_image.shape) == 3:
            target_gray = cv2.cvtColor(target_image, cv2.COLOR_RGB2GRAY)
        else:
            target_gray = target_image.copy()
        
        # Normaliser les images en niveaux de gris
        source_gray = source_gray / 255.0
        target_gray = target_gray / 255.0
        
        # Dimensions de sortie
        output_height, output_width = target_image.shape[:2]
        
        # Initialiser la texture de sortie
        if len(source_texture.shape) == 3:
            result = np.zeros((output_height, output_width, 3), dtype=np.uint8)
        else:
            result = np.zeros((output_height, output_width), dtype=np.uint8)
        
        # Réduire progressivement la taille des blocs à chaque itération
        for iteration in range(num_iterations):
            # Ajuster la taille du bloc pour cette itération
            block_size = self.block_size // (2 ** iteration)
            overlap = max(2, self.overlap // (2 ** iteration))
            
            # Ajuster alpha pour cette itération
            current_alpha = alpha - (iteration / (num_iterations - 1)) * (alpha - 0.1)
            
            # Calculer le nombre de blocs nécessaires
            n_blocks_h = int(np.ceil((output_height - overlap) / (block_size - overlap)))
            n_blocks_w = int(np.ceil((output_width - overlap) / (block_size - overlap)))
            
            # Pour chaque position de bloc
            for i in range(n_blocks_h):
                for j in range(n_blocks_w):
                    # Calculer la position du coin supérieur gauche du bloc
                    y = i * (block_size - overlap)
                    x = j * (block_size - overlap)
                    
                    # Taille du bloc actuel (peut être plus petit aux bords)
                    block_h = min(block_size, output_height - y)
                    block_w = min(block_size, output_width - x)
                    
                    # Région correspondante dans l'image cible
                    target_region = target_gray[y:y+block_h, x:x+block_w]
                    
                    # Déterminer les contraintes de chevauchement
                    constraints = {}
                    if i > 0:  # Il y a un bloc au-dessus
                        constraints['top'] = result[y:y+overlap, x:x+block_w]
                    if j > 0:  # Il y a un bloc à gauche
                        constraints['left'] = result[y:y+block_h, x:x+overlap]
                    
                    # Dans la première itération, nous avons besoin de trouver le meilleur bloc correspondant
                    # Dans les itérations suivantes, nous affinons également avec le résultat précédent
                    if iteration > 0:
                        constraints['previous'] = result[y:y+block_h, x:x+block_w]
                    
                    # Sélectionner un bloc approprié
                    block = self._find_matching_block_transfer(
                        source_texture, source_gray, target_region,
                        block_h, block_w, constraints, current_alpha
                    )
                    
                    # Appliquer la coupe minimale d'erreur si nécessaire
                    if i > 0 and j > 0:  # Chevauchement avec le bloc de gauche et celui du haut
                        block = self._minimum_error_boundary_cut_both(
                            result[y:y+overlap, x:x+block_w],  # Chevauchement haut
                            result[y:y+block_h, x:x+overlap],  # Chevauchement gauche
                            block, 
                            overlap
                        )
                    elif i > 0:  # Seulement chevauchement avec le bloc du haut
                        block = self._minimum_error_boundary_cut_horizontal(
                            result[y:y+overlap, x:x+block_w], 
                            block, 
                            overlap
                        )
                    elif j > 0:  # Seulement chevauchement avec le bloc de gauche
                        block = self._minimum_error_boundary_cut_vertical(
                            result[y:y+block_h, x:x+overlap], 
                            block, 
                            overlap
                        )
                    
                    # Placer le bloc dans l'image de sortie
                    result[y:y+block_h, x:x+block_w] = block
        
        return result
    
    def _find_matching_block_transfer(self, source_texture, source_gray, target_region, 
                                     block_h, block_w, constraints, alpha):
        """
        Trouver un bloc dans la texture source qui correspond à la fois aux contraintes
        et à la région cible pour le transfert de texture
        
        Args:
            source_texture (np.array): Texture source complète
            source_gray (np.array): Texture source en niveaux de gris normalisés
            target_region (np.array): Région cible en niveaux de gris normalisés
            block_h (int): Hauteur du bloc à extraire
            block_w (int): Largeur du bloc à extraire
            constraints (dict): Contraintes de chevauchement
            alpha (float): Poids pour l'équilibre entre texture et correspondance
            
        Returns:
            np.array: Bloc sélectionné
        """
        # Dimensions de la texture source
        input_h, input_w = source_texture.shape[:2]
        
        # Générer toutes les positions possibles pour le bloc
        valid_y = range(0, input_h - block_h + 1)
        valid_x = range(0, input_w - block_w + 1)
        
        # Calculer l'erreur pour chaque position possible
        min_error = float('inf')
        best_blocks = []
        
        for y in valid_y:
            for x in valid_x:
                texture_error = 0
                correspondence_error = 0
                
                # Calculer l'erreur de correspondance avec la région cible
                source_region = source_gray[y:y+block_h, x:x+block_w]
                correspondence_error = np.sum((source_region - target_region) ** 2)
                
                # Vérifier la contrainte du haut
                if 'top' in constraints:
                    top_constraint = constraints['top']
                    top_overlap = source_texture[y:y+self.overlap, x:x+block_w]
                    texture_error += np.sum((top_overlap - top_constraint) ** 2)
                
                # Vérifier la contrainte de gauche
                if 'left' in constraints:
                    left_constraint = constraints['left']
                    left_overlap = source_texture[y:y+block_h, x:x+self.overlap]
                    texture_error += np.sum((left_overlap - left_constraint) ** 2)
                
                # Vérifier la contrainte de la région précédente (pour les itérations > 0)
                if 'previous' in constraints:
                    prev_constraint = constraints['previous']
                    texture_error += np.sum((source_texture[y:y+block_h, x:x+block_w] - prev_constraint) ** 2) * 0.5
                
                # Normaliser les erreurs
                total_pixels = block_h * block_w
                correspondence_error /= total_pixels
                
                if 'top' in constraints and 'left' in constraints:
                    overlap_pixels = self.overlap * block_w + self.overlap * block_h - self.overlap**2
                    texture_error /= overlap_pixels
                elif 'top' in constraints:
                    texture_error /= (self.overlap * block_w)
                elif 'left' in constraints:
                    texture_error /= (self.overlap * block_h)
                elif 'previous' in constraints:
                    texture_error /= total_pixels
                else:
                    texture_error = 0  # Pas de contrainte de texture pour le premier bloc
                
                # Erreur totale pondérée
                total_error = alpha * texture_error + (1 - alpha) * correspondence_error
                
                # Mettre à jour le meilleur bloc
                if total_error < min_error:
                    min_error = total_error
                    best_blocks = [(y, x)]
                elif total_error < min_error * (1 + self.tolerance):
                    best_blocks.append((y, x))
        
        # Choisir aléatoirement parmi les meilleurs blocs
        y, x = random.choice(best_blocks)
        return source_texture[y:y+block_h, x:x+block_w].copy()
    
    def evaluate_texture(self, original_texture, synthesized_texture, patch_size=64):
        """
        Évaluer la qualité de la texture synthétisée en utilisant SSIM
        
        Args:
            original_texture (np.array): Texture originale
            synthesized_texture (np.array): Texture synthétisée
            patch_size (int): Taille des patches pour l'évaluation
            
        Returns:
            float: Score de similarité moyen (SSIM)
        """
        # Convertir en niveaux de gris si nécessaire
        if len(original_texture.shape) == 3:
            original_gray = cv2.cvtColor(original_texture, cv2.COLOR_RGB2GRAY)
        else:
            original_gray = original_texture.copy()
            
        if len(synthesized_texture.shape) == 3:
            synthesized_gray = cv2.cvtColor(synthesized_texture, cv2.COLOR_RGB2GRAY)
        else:
            synthesized_gray = synthesized_texture.copy()
        
        # Dimensions des textures
        orig_h, orig_w = original_gray.shape
        synth_h, synth_w = synthesized_gray.shape
        
        # Extraire des patches aléatoires de la texture originale
        num_patches = 10
        original_patches = []
        
        for _ in range(num_patches):
            if orig_h <= patch_size or orig_w <= patch_size:
                original_patches.append(original_gray)
            else:
                y = np.random.randint(0, orig_h - patch_size)
                x = np.random.randint(0, orig_w - patch_size)
                original_patches.append(original_gray[y:y+patch_size, x:x+patch_size])
        
        # Extraire des patches aléatoires de la texture synthétisée
        synthesized_patches = []
        
        for _ in range(num_patches * 2):
            if synth_h <= patch_size or synth_w <= patch_size:
                synthesized_patches.append(synthesized_gray)
            else:
                y = np.random.randint(0, synth_h - patch_size)
                x = np.random.randint(0, synth_w - patch_size)
                synthesized_patches.append(synthesized_gray[y:y+patch_size, x:x+patch_size])
        
        # Calculer les scores SSIM pour chaque paire de patches
        ssim_scores = []
        
        for orig_patch in original_patches:
            for synth_patch in synthesized_patches:
                # Redimensionner les patches si nécessaire
                if orig_patch.shape != synth_patch.shape:
                    orig_patch = cv2.resize(orig_patch, (patch_size, patch_size))
                    synth_patch = cv2.resize(synth_patch, (patch_size, patch_size))
                
                score = ssim(orig_patch, synth_patch, data_range=255)
                ssim_scores.append(score)
        
        # Retourner le score SSIM moyen
        return np.mean(ssim_scores)

# Fonction principale pour tester l'algorithme
def main():
    """
    Fonction principale pour tester l'algorithme avec différentes textures
    et évaluer les résultats
    """
    # Créer un dossier pour les résultats
    import os
    os.makedirs("results", exist_ok=True)
    
    # Liste des chemins de texture (remplacer par vos chemins)
    texture_paths = [
        "texture_input.jpg",
        # Ajoutez d'autres textures ici
    ]
    
    # Liste des chemins d'image cible pour le transfert (remplacer par vos chemins)
    target_paths = [
        "target_image.jpg",
        # Ajoutez d'autres images cibles ici
    ]
    
    # 1. Tester la synthèse sur différentes textures
    print("=== TEST DE SYNTHÈSE DE TEXTURE ===")
    for path in texture_paths:
        print(f"\nTraitement de {path}...")
        
        # Charger la texture
        input_texture = cv2.imread(path)
        if input_texture is None:
            print(f"Erreur: Impossible de charger l'image {path}")
            continue
        
        input_texture = cv2.cvtColor(input_texture, cv2.COLOR_BGR2RGB)
        
        # Tester différentes tailles de blocs
        for block_size in [16, 32, 48]:
            overlap = block_size // 6  # Environ 1/6 de la taille du bloc
            
            print(f"Synthèse avec taille de bloc={block_size}, chevauchement={overlap}")
            
            # Créer l'instance de l'algorithme
            quilting = ImageQuilting(block_size=block_size, overlap=overlap, tolerance=0.1)
            
            # Synthétiser une nouvelle texture
            start_time = time.time()
            output_texture = quilting.synthesize_texture(
                input_texture, 
                output_height=input_texture.shape[0] * 2, 
                output_width=input_texture.shape[1] * 2
            )
            end_time = time.time()
            
            # Évaluer la qualité
            ssim_score = quilting.evaluate_texture(input_texture, output_texture)
            
            # Afficher les résultats
            print(f"  Temps de synthèse: {end_time - start_time:.2f} secondes")
            print(f"  Score SSIM: {ssim_score:.4f}")
            
            # Sauvegarder le résultat
            texture_name = os.path.splitext(os.path.basename(path))[0]
            output_filename = f"results/synth_{texture_name}_b{block_size}.png"
            plt.figure(figsize=(12, 6))
            
            plt.subplot(1, 2, 1)
            plt.imshow(input_texture)
            plt.title("Texture d'entrée")
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(output_texture)
            plt.title(f"Synthèse (b={block_size}, SSIM={ssim_score:.4f})")
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(output_filename)
            plt.close()
    
    # 2. Tester le transfert de texture
    print("\n=== TEST DE TRANSFERT DE TEXTURE ===")
    
    # Utiliser la première texture disponible comme source
    source_texture = None
    for path in texture_paths:
        input_texture = cv2.imread(path)
        if input_texture is not None:
            source_texture = cv2.cvtColor(input_texture, cv2.COLOR_BGR2RGB)
            print(f"Utilisation de {path} comme texture source")
            break
    
    if source_texture is None:
        print("Aucune texture source valide trouvée. Le transfert de texture ne sera pas testé.")
        return
    
    # Tester le transfert sur différentes images cibles
    for target_path in target_paths:
        print(f"\nTransfert vers {target_path}...")
        
        # Charger l'image cible
        target_image = cv2.imread(target_path)
        if target_image is None:
            print(f"Erreur: Impossible de charger l'image cible {target_path}")
            continue
        
        target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
        
        # Créer l'instance de l'algorithme avec des paramètres par défaut
        quilting = ImageQuilting(block_size=32, overlap=6, tolerance=0.1)
        
        # Transférer la texture
        start_time = time.time()
        transferred_texture = quilting.texture_transfer(
            source_texture, 
            target_image,
            num_iterations=3,
            alpha=0.8
        )
        end_time = time.time()
        
        print(f"  Temps de transfert: {end_time - start_time:.2f} secondes")
        
        # Sauvegarder le résultat
        texture_name = os.path.splitext(os.path.basename(texture_paths[0]))[0]
        target_name = os.path.splitext(os.path.basename(target_path))[0]
        output_filename = f"results/transfer_{texture_name}_to_{target_name}.png"
        
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(source_texture)
        plt.title("Texture source")
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(target_image)
        plt.title("Image cible")
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(transferred_texture)
        plt.title("Texture transférée")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_filename)
        plt.close()

# Exemple d'utilisation
if __name__ == "__main__":
    # Exemple simple
    # Charger une texture d'entrée
    input_image_path = "texture_input.jpg"  # Remplacer par votre chemin
    input_texture = cv2.imread(input_image_path)
    if input_texture is None:
        print(f"Erreur: Impossible de charger l'image {input_image_path}")
        print("Veuillez spécifier un chemin valide ou exécuter la fonction main() pour tester plusieurs configurations")
    else:
        input_texture = cv2.cvtColor(input_texture, cv2.COLOR_BGR2RGB)  # Convertir en RGB
    
    # Définir les paramètres
    block_size = 32  # Taille des blocs
    overlap = 6      # Taille du chevauchement (1/6 de la taille du bloc selon le papier)
    tolerance = 0.1  # Tolérance d'erreur
    
    # Créer l'instance de l'algorithme
    quilting = ImageQuilting(block_size=block_size, overlap=overlap, tolerance=tolerance)
    
    # Synthétiser une nouvelle texture
    start_time = time.time()
    output_texture = quilting.synthesize_texture(
        input_texture, 
        output_height=input_texture.shape[0] * 2, 
        output_width=input_texture.shape[1] * 2
    )
    end_time = time.time()
    
    print(f"Synthèse de texture terminée en {end_time - start_time:.2f} secondes")
    
    # Évaluer la qualité de la texture synthétisée
    ssim_score = quilting.evaluate_texture(input_texture, output_texture)
    print(f"Score SSIM: {ssim_score:.4f}")
    
    # Afficher les résultats
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(input_texture)
    plt.title("Texture d'entrée")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(output_texture)
    plt.title("Texture synthétisée")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("texture_synthesis_result.png")
    plt.show()
    
    # Essayer de charger une image cible pour le transfert de texture
    try:
        target_image_path = "target_image.jpg"  # Remplacer par votre chemin
        target_image = cv2.imread(target_image_path)
        
        if target_image is not None:
            target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)  # Convertir en RGB
            
            # Transférer la texture
            print("\nDémarrage du transfert de texture...")
            start_time = time.time()
            transferred_texture = quilting.texture_transfer(
                input_texture, 
                target_image,
                num_iterations=3,  # Nombre d'itérations recommandé dans le papier
                alpha=0.8          # Alpha initial recommandé dans le papier
            )
            end_time = time.time()
            
            print(f"Transfert de texture terminé en {end_time - start_time:.2f} secondes")
            
            # Afficher les résultats du transfert
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.imshow(input_texture)
            plt.title("Texture source")
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(target_image)
            plt.title("Image cible")
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(transferred_texture)
            plt.title("Texture transférée")
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig("texture_transfer_result.png")
            plt.show()
        else:
            print(f"\nAvertissement: Impossible de charger l'image cible {target_image_path} pour le transfert de texture")
            print("Le transfert de texture n'a pas été effectué")
    except Exception as e:
        print(f"\nErreur lors du transfert de texture: {e}")
        print("Le transfert de texture n'a pas été effectué")
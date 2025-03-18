import math
import numpy as np


def inertia_factory(feat_metric, sample_metric):
    def feat_inner(u, v):
        """Produit scalaire entre `u` et `v`."""

        return (u[None, :] @ feat_metric @ v[:, None]).item()

    def inertia_point(X, p=None):
        """Calcule l'inertie de `X` par rapport au point `p`. Si le point `p`
        n'est pas fourni prendre le centre de gravité.
        """

        if p is None:
            # Calcul du centre de gravité pondéré par `sample_metric`.
            # On utilisera `np.sum` avec le bon argument `axis`.
            p = ...

        # Linéarisation au cas où...
        p = p.squeeze()

        # Centrer par rapport à `p`. On pourra utiliser le broadcasting.
        Xp = ...

        # Utiliser la formule du cours avec la trace.
        inertia = ...

        return inertia

    def inertia_axe(X, v, p=None):
        """Calcule l'inertie de `X` par rapport à la droite pointée par `p` et
        de direction `v`. Si `p` n'est pas fourni, on prend la droite
        qui passe par l'origine (p=0).
        """

        if p is None:
            p = np.zeros(2)

        # Linéarisation au cas où...
        p = p.squeeze()

        def dist2_axe_point(v, p, q):
            """Distance au carré d'un point `q` à une droite repérée par un
            vecteur `v` et un point `p`.
            """

            # Normalisation du vecteur `v` à l'aide de `feat_inner`
            v = ...
            v = ...

            # Calculer le vecteur perpendiculaire à la droite
            # d'origine q et dont l'extrémité appartient à la droite.
            q_projq = ...

            # Retourner la norme au carré de `q_projq` en utilisant
            # `feat_inner`.
            return ...

        # Matrice de toutes les distances au carré à la droite. On
        # utilisera `dist_axe_point`.
        dists2 = ...

        # Somme pondérée en fonction de `sample_metric`
        inertia = ...

        return inertia

    return feat_inner, inertia_point, inertia_axe


if __name__ == '__main__':
    # Exemple d'utilisation de `inertia_factory`.
    n, p = 100, 2

    # Matrice M
    feat_metric = np.eye(p)

    # Matrice D_p
    sample_metric = 1 / n * np.eye(n)

    # Création de `inertia_point` et `inertia_axe` qui calculent
    # l'inertie par rapport à un point et l'inertie par rapport à un
    # axe avec les métriques fournies.
    feat_inner, inertia_point, inertia_axe = inertia_factory(feat_metric, sample_metric)

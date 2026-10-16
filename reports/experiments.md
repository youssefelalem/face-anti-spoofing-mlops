# Experiences - choix de la configuration (sans le jeu de test)

Entrainement : session 01 (1744 images) - validation : session 02 (1747 images).

| Configuration | Normalisation photo | Aug. forte | Meilleure epoque | AUC val | EER val | AUC val eclairage inverse | EER val eclairage inverse |
|---|---|---|---|---|---|---|---|
| E1-baseline | non | non | 3 | 0.9983 | 1.77 % | 0.9976 | 2.80 % |
| E2-photo-norm | oui | non | 3 | 0.9892 | 5.95 % | 0.9827 | 7.61 % |
| E3-photo-norm-strong-aug | oui | oui | 2 | 0.9842 | 7.10 % | 0.9775 | 8.76 % |

Regle de choix : la meilleure EER sur la validation a eclairage inverse (robustesse), a performance normale comparable.

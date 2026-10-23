/**
 * stats.ts : toutes les mathématiques de l'atelier, écrites à la main
 * (aucune bibliothèque de machine learning).
 *
 *   Partie 1 : décrire les données     (moyenne, médiane, écart-type, corrélation)
 *   Partie 2 : construire et évaluer   (régression linéaire, MSE, R²)
 *   Partie 3 : préparer les données    (hasard reproductible, jeu synthétique, découpage)
 *   Partie 4 : régression polynomiale  (moindres carrés + élimination de Gauss)
 *   Partie 5 : descente de gradient    (la machine qui apprend pas à pas)
 */

/** Vérifie que deux listes forment des paires : chaque x a son y. */
function verifierPaires(xs: number[], ys: number[]): void {
    if (xs.length !== ys.length) {
        throw new Error(`xs (${xs.length}) et ys (${ys.length}) n'ont pas la même longueur`);
    }
}

// ---------------------------------------------------------------------
// PARTIE 1 : DÉCRIRE LES DONNÉES
// ---------------------------------------------------------------------

/** Moyenne : x̄ = Σx / n, le point d'équilibre des données. */
export function mean(xs: number[]): number {
    if (xs.length === 0) throw new Error("mean : tableau vide");
    let somme = 0;
    for (const x of xs) somme += x;
    return somme / xs.length;
}

/** Médiane : la valeur du milieu après tri. Résiste aux valeurs aberrantes. */
export function median(xs: number[]): number {
    if (xs.length === 0) throw new Error("median : tableau vide");
    // On trie une copie, car sort() modifie le tableau d'origine.
    // (a, b) => a - b force un tri numérique (sinon [10, 9, 1] devient [1, 10, 9]).
    const tri = [...xs].sort((a, b) => a - b);
    const milieu = Math.floor(tri.length / 2);
    return tri.length % 2 === 1
        ? tri[milieu]                            // nombre impair : la valeur centrale
        : (tri[milieu - 1] + tri[milieu]) / 2;   // nombre pair : moyenne des 2 centrales
}

/** Écart-type d'échantillon : s = √( Σ(x − x̄)² / (n − 1) ). */
export function sd(xs: number[]): number {
    if (xs.length < 2) throw new Error("sd : il faut au moins 2 valeurs");
    const m = mean(xs);
    let sommeCarres = 0;
    for (const x of xs) sommeCarres += (x - m) ** 2;
    // n − 1 (correction de Bessel) : x̄ est lui-même estimé sur l'échantillon.
    return Math.sqrt(sommeCarres / (xs.length - 1));
}

/** Corrélation de Pearson : r = Sxy / √(Sxx · Syy), toujours entre −1 et 1. */
export function pearson(xs: number[], ys: number[]): number {
    verifierPaires(xs, ys);
    if (xs.length < 2) throw new Error("pearson : il faut au moins 2 points");
    const mx = mean(xs);
    const my = mean(ys);
    let sxy = 0, sxx = 0, syy = 0;
    for (let i = 0; i < xs.length; i++) {
        const dx = xs[i] - mx;
        const dy = ys[i] - my;
        sxy += dx * dy; // x et y montent-ils ensemble ?
        sxx += dx * dx; // dispersion de x
        syy += dy * dy; // dispersion de y
    }
    if (sxx === 0 || syy === 0) return 0; // aucune variation : évite une division par 0
    return sxy / Math.sqrt(sxx * syy);
}

// ---------------------------------------------------------------------
// PARTIE 2 : CONSTRUIRE ET ÉVALUER UN MODÈLE
// ---------------------------------------------------------------------

/** Une droite : ŷ = pente · x + ordonnee. */
export interface Droite {
    pente: number;    // de combien y monte quand x augmente de 1
    ordonnee: number; // la valeur de y quand x = 0
}

/** Régression linéaire exacte : pente = Sxy / Sxx, ordonnee = ȳ − pente · x̄. */
export function linreg(xs: number[], ys: number[]): Droite {
    verifierPaires(xs, ys);
    if (xs.length < 2) throw new Error("linreg : il faut au moins 2 points");
    const mx = mean(xs);
    const my = mean(ys);
    let sxy = 0, sxx = 0;
    for (let i = 0; i < xs.length; i++) {
        const dx = xs[i] - mx;
        sxy += dx * (ys[i] - my);
        sxx += dx * dx;
    }
    // Si tous les x sont identiques, la droite serait verticale (pente infinie).
    if (sxx === 0) throw new Error("linreg : tous les x sont identiques");
    const pente = sxy / sxx;
    return { pente, ordonnee: my - pente * mx }; // la droite passe par (x̄, ȳ)
}

/** Prédit y à partir d'une droite. */
export function predireLineaire(d: Droite, x: number): number {
    return d.pente * x + d.ordonnee;
}

/** Erreur quadratique moyenne : MSE = Σ(y − ŷ)² / n (0 = parfait). */
export function mse(ys: number[], yPred: number[]): number {
    verifierPaires(ys, yPred);
    if (ys.length === 0) throw new Error("mse : tableau vide");
    let somme = 0;
    for (let i = 0; i < ys.length; i++) {
        const residu = ys[i] - yPred[i]; // l'erreur sur le point i
        somme += residu * residu;
    }
    return somme / ys.length;
}

/**
 * Coefficient de détermination : R² = 1 − SSres / SStot.
 * Compare le modèle au modèle « paresseux » qui prédit toujours la moyenne :
 *   R² = 1  modèle parfait
 *   R² = 0  pas mieux que la moyenne
 *   R² < 0  pire que la moyenne
 */
export function r2(ys: number[], yPred: number[]): number {
    verifierPaires(ys, yPred);
    if (ys.length === 0) throw new Error("r2 : tableau vide");
    const my = mean(ys);
    let ssRes = 0; // erreur de notre modèle
    let ssTot = 0; // erreur du modèle paresseux
    for (let i = 0; i < ys.length; i++) {
        ssRes += (ys[i] - yPred[i]) ** 2;
        ssTot += (ys[i] - my) ** 2;
    }
    if (ssTot === 0) return ssRes === 0 ? 1 : 0; // tous les y sont identiques
    return 1 - ssRes / ssTot;
}

// ---------------------------------------------------------------------
// PARTIE 3 : PRÉPARER LES DONNÉES
// ---------------------------------------------------------------------

/**
 * Générateur pseudo-aléatoire « mulberry32 ».
 * Reçoit une graine et renvoie une fonction qui donne un nombre dans [0, 1)
 * à chaque appel. Même graine = même suite de nombres, toujours.
 *
 * Un état interne de 32 bits avance d'une constante à chaque appel, puis ses
 * bits sont brassés (décalages >>>, XOR ^, multiplications Math.imul) pour que
 * le résultat paraisse aléatoire. « >>> 0 » garde un entier positif de 32 bits.
 */
export function creerAleatoire(graine: number): () => number {
    let etat = graine >>> 0;
    return function (): number {
        etat = (etat + 0x6d2b79f5) >>> 0;
        let t = etat;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296; // ÷ 2³² : résultat dans [0, 1)
    };
}

/**
 * Bruit gaussien (loi normale de moyenne 0 et d'écart-type 1), obtenu par la
 * transformation de Box-Muller : deux nombres uniformes donnent une
 * « courbe en cloche ». Dans la nature, les petites erreurs sont fréquentes
 * et les grosses sont rares.
 */
export function bruitGaussien(alea: () => number): number {
    const u1 = 1 - alea(); // dans (0, 1] : évite log(0) = −∞
    const u2 = alea();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

/** Un point du graphique. */
export interface Point {
    x: number;
    y: number;
}

/**
 * Jeu synthétique : y = 2x + 1 + bruit.
 *   n          : nombre de points
 *   graine     : une autre graine donne un nouveau tirage
 *   ecartBruit : intensité du bruit (écart-type)
 */
export function genererSynthetique(n = 40, graine = 1, ecartBruit = 2): Point[] {
    const alea = creerAleatoire(graine);
    const points: Point[] = [];
    for (let i = 0; i < n; i++) {
        const x = alea() * 10;                                  // x au hasard entre 0 et 10
        const y = 2 * x + 1 + ecartBruit * bruitGaussien(alea); // la règle cachée + du bruit
        points.push({ x, y });
    }
    return points;
}

/**
 * Découpage entraînement / test :
 *   1) on mélange une copie des points (algorithme de Fisher-Yates) ;
 *   2) les 70 % premiers servent à l'entraînement, le reste au test.
 * La graine rend le mélange reproductible.
 */
export function decouperEntrainementTest(
    points: Point[],
    ratioEntrainement = 0.7,
    graine = 7
): { entrainement: Point[]; test: Point[] } {
    const alea = creerAleatoire(graine);
    const melange = [...points]; // copie : les données d'origine restent intactes

    // Fisher-Yates : de la fin vers le début, on échange chaque case avec une
    // case choisie au hasard parmi celles qui restent. Tous les ordres
    // possibles ont la même probabilité.
    for (let i = melange.length - 1; i > 0; i--) {
        const j = Math.floor(alea() * (i + 1)); // indice au hasard entre 0 et i
        [melange[i], melange[j]] = [melange[j], melange[i]];
    }

    const nEntrainement = Math.round(melange.length * ratioEntrainement);
    return {
        entrainement: melange.slice(0, nEntrainement),
        test: melange.slice(nEntrainement),
    };
}

// ---------------------------------------------------------------------
// PARTIE 4 : RÉGRESSION POLYNOMIALE
// ---------------------------------------------------------------------

/**
 * Un polynôme ajusté. Les coefficients s'appliquent à z, la version
 * normalisée de x : z = (x − centre) / echelle, qui tombe dans [−1, 1].
 */
export interface Polynome {
    coefs: number[]; // [c0, c1, ..., cd] : ŷ = c0 + c1·z + c2·z² + ...
    centre: number;  // milieu de la plage des x d'entraînement
    echelle: number; // demi-largeur de cette plage
}

/**
 * Résout le système A · c = b par élimination de Gauss avec pivot partiel.
 * A : matrice carrée n × n ; b : vecteur de taille n. Renvoie c.
 */
export function resoudreGauss(A: number[][], b: number[]): number[] {
    const n = b.length;
    // Matrice augmentée [A | b], construite sur une copie pour ne pas modifier A et b.
    const M = A.map((ligne, i) => [...ligne, b[i]]);

    // 1) Descente : on crée des zéros sous la diagonale, colonne par colonne.
    for (let col = 0; col < n; col++) {
        // Pivot partiel : on prend la ligne au plus grand |coefficient| (plus stable).
        let pivot = col;
        for (let lig = col + 1; lig < n; lig++) {
            if (Math.abs(M[lig][col]) > Math.abs(M[pivot][col])) pivot = lig;
        }
        if (Math.abs(M[pivot][col]) < 1e-12) {
            throw new Error("Gauss : système singulier (pas de solution unique)");
        }
        [M[col], M[pivot]] = [M[pivot], M[col]];

        for (let lig = col + 1; lig < n; lig++) {
            const facteur = M[lig][col] / M[col][col];
            for (let k = col; k <= n; k++) {
                M[lig][k] -= facteur * M[col][k];
            }
        }
    }

    // 2) Remontée : on calcule les inconnues de la dernière à la première.
    const solution: number[] = new Array(n).fill(0);
    for (let lig = n - 1; lig >= 0; lig--) {
        let somme = M[lig][n];
        for (let k = lig + 1; k < n; k++) {
            somme -= M[lig][k] * solution[k];
        }
        solution[lig] = somme / M[lig][lig];
    }
    return solution;
}

/**
 * Régression polynomiale de degré `degre` par les moindres carrés :
 * on construit le système (XᵀX) c = Xᵀy, puis on le résout avec Gauss.
 */
export function polyfit(xs: number[], ys: number[], degre: number): Polynome {
    verifierPaires(xs, ys);
    if (xs.length <= degre) {
        throw new Error(`polyfit : il faut plus de ${degre} points pour un degré ${degre}`);
    }

    // Normalisation de x dans [−1, 1] pour éviter des puissances gigantesques.
    const min = Math.min(...xs);
    const max = Math.max(...xs);
    const centre = (min + max) / 2;
    const echelle = (max - min) / 2 || 1; // « || 1 » évite ÷ 0 si tous les x sont égaux

    const m = degre + 1; // nombre d'inconnues
    const XtX = Array.from({ length: m }, () => new Array(m).fill(0));
    const Xty = new Array(m).fill(0);

    for (let i = 0; i < xs.length; i++) {
        const z = (xs[i] - centre) / echelle;
        // Puissances de z de 0 à 2·degre : [1, z, z², ..., z^(2d)]
        const puiss = [1];
        for (let p = 1; p <= 2 * degre; p++) puiss.push(puiss[p - 1] * z);

        for (let j = 0; j < m; j++) {
            Xty[j] += puiss[j] * ys[i];    // Xᵀy[j]    = Σ z^j · y
            for (let k = 0; k < m; k++) {
                XtX[j][k] += puiss[j + k]; // XᵀX[j][k] = Σ z^(j+k)
            }
        }
    }

    return { coefs: resoudreGauss(XtX, Xty), centre, echelle };
}

/**
 * Prédit y avec un polynôme, par le schéma de Horner :
 * c0 + c1·z + c2·z² + c3·z³ = c0 + z·(c1 + z·(c2 + z·c3)).
 * Moins de multiplications et plus précis que de calculer chaque puissance.
 */
export function predirePoly(p: Polynome, x: number): number {
    const z = (x - p.centre) / p.echelle; // même normalisation qu'à l'entraînement
    let resultat = 0;
    for (let j = p.coefs.length - 1; j >= 0; j--) {
        resultat = resultat * z + p.coefs[j];
    }
    return resultat;
}

// ---------------------------------------------------------------------
// PARTIE 5 : DESCENTE DE GRADIENT
// ---------------------------------------------------------------------

/** Tout ce qu'il faut pour afficher l'apprentissage à l'écran. */
export interface ResultatDescente {
    pente: number;    // pente finale, dans les unités d'origine de x
    ordonnee: number; // ordonnée finale, dans les unités d'origine
    pertes: number[]; // la perte à chaque itération (courbe de perte)
    etapes: Droite[]; // la droite à chaque itération (animation)
    diverge: boolean; // true si l'apprentissage a échoué (pas trop grand)
}

/**
 * Régression linéaire par descente de gradient.
 *   xs, ys       : les données d'entraînement
 *   pas          : le taux d'apprentissage α
 *   nbIterations : nombre de pas effectués
 *   normaliser   : centrer-réduire x avant d'apprendre (false pour voir l'effet sans)
 */
export function gradientDescentLinear(
    xs: number[],
    ys: number[],
    pas = 0.1,
    nbIterations = 100,
    normaliser = true
): ResultatDescente {
    verifierPaires(xs, ys);
    if (xs.length < 2) throw new Error("gradientDescentLinear : il faut au moins 2 points");

    // 1) Normalisation : z = (x − x̄) / s ; sans normalisation, z = x.
    const moyX = normaliser ? mean(xs) : 0;
    const ecartX = normaliser ? sd(xs) || 1 : 1; // « || 1 » évite ÷ 0
    const zs = xs.map((x) => (x - moyX) / ecartX);
    const n = xs.length;

    // Convertit (w, b) de l'espace normalisé vers les unités d'origine.
    const versOrigine = (w: number, b: number): Droite => ({
        pente: w / ecartX,
        ordonnee: b - (w * moyX) / ecartX,
    });

    // 2) Point de départ : une droite plate à 0 (la machine ne sait encore rien).
    let w = 0;
    let b = 0;
    const pertes: number[] = [];
    const etapes: Droite[] = [];

    for (let iter = 0; iter < nbIterations; iter++) {
        // a) Prédire, mesurer l'erreur et accumuler les gradients.
        let perte = 0;
        let gradW = 0;
        let gradB = 0;
        for (let i = 0; i < n; i++) {
            const erreur = w * zs[i] + b - ys[i]; // ŷ − y
            perte += erreur * erreur;
            gradW += erreur * zs[i];
            gradB += erreur;
        }
        perte /= n;              // MSE
        gradW = (2 / n) * gradW; // ∂L/∂w
        gradB = (2 / n) * gradB; // ∂L/∂b

        // b) On mémorise l'état avant le pas (pour la courbe et l'animation).
        pertes.push(perte);
        etapes.push(versOrigine(w, b));

        // c) Si les nombres explosent (Infinity ou NaN), inutile de continuer.
        if (!Number.isFinite(perte)) break;

        // d) Le pas : on descend dans le sens opposé au gradient.
        w -= pas * gradW;
        b -= pas * gradB;
    }

    // 3) A-t-on divergé ? (perte finale infinie, ou plus grande qu'au départ)
    const premiere = pertes[0];
    const derniere = pertes[pertes.length - 1];
    const diverge = !Number.isFinite(derniere) || derniere > premiere;

    const finale = versOrigine(w, b);
    return { pente: finale.pente, ordonnee: finale.ordonnee, pertes, etapes, diverge };
}

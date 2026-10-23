/** Statistiques, régressions et descente de gradient, sans bibliothèque de ML. */

function verifierPaires(xs: number[], ys: number[]): void {
    if (xs.length !== ys.length) {
        throw new Error(`xs (${xs.length}) et ys (${ys.length}) n'ont pas la même longueur`);
    }
}

// ---------------------------------------------------------------------
// PARTIE 1 : DÉCRIRE LES DONNÉES
// ---------------------------------------------------------------------

/** Moyenne : x̄ = Σx / n */
export function mean(xs: number[]): number {
    if (xs.length === 0) throw new Error("mean : tableau vide");
    let somme = 0;
    for (const x of xs) somme += x;
    return somme / xs.length;
}

/** Médiane : valeur centrale après tri. */
export function median(xs: number[]): number {
    if (xs.length === 0) throw new Error("median : tableau vide");
    // Copie (sort() modifie le tableau) et tri numérique (sinon tri alphabétique).
    const tri = [...xs].sort((a, b) => a - b);
    const milieu = Math.floor(tri.length / 2);
    return tri.length % 2 === 1
        ? tri[milieu]
        : (tri[milieu - 1] + tri[milieu]) / 2;
}

/** Écart-type d'échantillon : s = √( Σ(x − x̄)² / (n − 1) ) */
export function sd(xs: number[]): number {
    if (xs.length < 2) throw new Error("sd : il faut au moins 2 valeurs");
    const m = mean(xs);
    let sommeCarres = 0;
    for (const x of xs) sommeCarres += (x - m) ** 2;
    // n − 1 : correction de Bessel
    return Math.sqrt(sommeCarres / (xs.length - 1));
}

/** Corrélation de Pearson : r = Sxy / √(Sxx · Syy), entre −1 et 1 */
export function pearson(xs: number[], ys: number[]): number {
    verifierPaires(xs, ys);
    if (xs.length < 2) throw new Error("pearson : il faut au moins 2 points");
    const mx = mean(xs);
    const my = mean(ys);
    let sxy = 0, sxx = 0, syy = 0;
    for (let i = 0; i < xs.length; i++) {
        const dx = xs[i] - mx;
        const dy = ys[i] - my;
        sxy += dx * dy;
        sxx += dx * dx;
        syy += dy * dy;
    }
    if (sxx === 0 || syy === 0) return 0;
    return sxy / Math.sqrt(sxx * syy);
}

// ---------------------------------------------------------------------
// PARTIE 2 : CONSTRUIRE ET ÉVALUER UN MODÈLE
// ---------------------------------------------------------------------

/** ŷ = pente · x + ordonnee */
export interface Droite {
    pente: number;
    ordonnee: number;
}

/** Régression linéaire exacte : pente = Sxy / Sxx, ordonnee = ȳ − pente · x̄ */
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
    if (sxx === 0) throw new Error("linreg : tous les x sont identiques");
    const pente = sxy / sxx;
    return { pente, ordonnee: my - pente * mx };
}

export function predireLineaire(d: Droite, x: number): number {
    return d.pente * x + d.ordonnee;
}

/** Erreur quadratique moyenne : MSE = Σ(y − ŷ)² / n */
export function mse(ys: number[], yPred: number[]): number {
    verifierPaires(ys, yPred);
    if (ys.length === 0) throw new Error("mse : tableau vide");
    let somme = 0;
    for (let i = 0; i < ys.length; i++) {
        const residu = ys[i] - yPred[i];
        somme += residu * residu;
    }
    return somme / ys.length;
}

/** R² = 1 − SSres / SStot (1 : parfait, 0 : comme la moyenne, < 0 : pire) */
export function r2(ys: number[], yPred: number[]): number {
    verifierPaires(ys, yPred);
    if (ys.length === 0) throw new Error("r2 : tableau vide");
    const my = mean(ys);
    let ssRes = 0;
    let ssTot = 0;
    for (let i = 0; i < ys.length; i++) {
        ssRes += (ys[i] - yPred[i]) ** 2;
        ssTot += (ys[i] - my) ** 2;
    }
    if (ssTot === 0) return ssRes === 0 ? 1 : 0;
    return 1 - ssRes / ssTot;
}

// ---------------------------------------------------------------------
// PARTIE 3 : PRÉPARER LES DONNÉES
// ---------------------------------------------------------------------

/** Générateur pseudo-aléatoire mulberry32 : même graine, même suite dans [0, 1). */
export function creerAleatoire(graine: number): () => number {
    let etat = graine >>> 0;
    return function (): number {
        etat = (etat + 0x6d2b79f5) >>> 0;
        let t = etat;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

/** Bruit gaussien N(0, 1) par la transformation de Box-Muller. */
export function bruitGaussien(alea: () => number): number {
    const u1 = 1 - alea(); // dans (0, 1] : évite log(0) = −∞
    const u2 = alea();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

export interface Point {
    x: number;
    y: number;
}

/** Jeu synthétique : y = 2x + 1 + bruit gaussien. */
export function genererSynthetique(n = 40, graine = 1, ecartBruit = 2): Point[] {
    const alea = creerAleatoire(graine);
    const points: Point[] = [];
    for (let i = 0; i < n; i++) {
        const x = alea() * 10;
        const y = 2 * x + 1 + ecartBruit * bruitGaussien(alea);
        points.push({ x, y });
    }
    return points;
}

/** Mélange reproductible (Fisher-Yates), puis découpage entraînement / test. */
export function decouperEntrainementTest(
    points: Point[],
    ratioEntrainement = 0.7,
    graine = 7
): { entrainement: Point[]; test: Point[] } {
    const alea = creerAleatoire(graine);
    const melange = [...points];

    for (let i = melange.length - 1; i > 0; i--) {
        const j = Math.floor(alea() * (i + 1));
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

/** Coefficients appliqués à z = (x − centre) / echelle, dans [−1, 1]. */
export interface Polynome {
    coefs: number[];
    centre: number;
    echelle: number;
}

/** Résout A · c = b par élimination de Gauss avec pivot partiel. */
export function resoudreGauss(A: number[][], b: number[]): number[] {
    const n = b.length;
    const M = A.map((ligne, i) => [...ligne, b[i]]);

    for (let col = 0; col < n; col++) {
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

    // Remontée
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

/** Régression polynomiale par les moindres carrés : (XᵀX) c = Xᵀy. */
export function polyfit(xs: number[], ys: number[], degre: number): Polynome {
    verifierPaires(xs, ys);
    if (xs.length <= degre) {
        throw new Error(`polyfit : il faut plus de ${degre} points pour un degré ${degre}`);
    }

    // x ramené dans [−1, 1] pour éviter des puissances gigantesques.
    const min = Math.min(...xs);
    const max = Math.max(...xs);
    const centre = (min + max) / 2;
    const echelle = (max - min) / 2 || 1;

    const m = degre + 1;
    const XtX = Array.from({ length: m }, () => new Array(m).fill(0));
    const Xty = new Array(m).fill(0);

    for (let i = 0; i < xs.length; i++) {
        const z = (xs[i] - centre) / echelle;
        const puiss = [1];
        for (let p = 1; p <= 2 * degre; p++) puiss.push(puiss[p - 1] * z);

        for (let j = 0; j < m; j++) {
            Xty[j] += puiss[j] * ys[i];
            for (let k = 0; k < m; k++) {
                XtX[j][k] += puiss[j + k];
            }
        }
    }

    return { coefs: resoudreGauss(XtX, Xty), centre, echelle };
}

/** Évalue le polynôme par le schéma de Horner. */
export function predirePoly(p: Polynome, x: number): number {
    const z = (x - p.centre) / p.echelle;
    let resultat = 0;
    for (let j = p.coefs.length - 1; j >= 0; j--) {
        resultat = resultat * z + p.coefs[j];
    }
    return resultat;
}

// ---------------------------------------------------------------------
// PARTIE 5 : DESCENTE DE GRADIENT
// ---------------------------------------------------------------------

export interface ResultatDescente {
    pente: number;
    ordonnee: number;
    pertes: number[];
    etapes: Droite[];
    diverge: boolean;
}

/** Régression linéaire par descente de gradient (x centré-réduit si normaliser). */
export function gradientDescentLinear(
    xs: number[],
    ys: number[],
    pas = 0.1,
    nbIterations = 100,
    normaliser = true
): ResultatDescente {
    verifierPaires(xs, ys);
    if (xs.length < 2) throw new Error("gradientDescentLinear : il faut au moins 2 points");

    const moyX = normaliser ? mean(xs) : 0;
    const ecartX = normaliser ? sd(xs) || 1 : 1;
    const zs = xs.map((x) => (x - moyX) / ecartX);
    const n = xs.length;

    // (w, b) normalisés → unités d'origine
    const versOrigine = (w: number, b: number): Droite => ({
        pente: w / ecartX,
        ordonnee: b - (w * moyX) / ecartX,
    });

    let w = 0;
    let b = 0;
    const pertes: number[] = [];
    const etapes: Droite[] = [];

    for (let iter = 0; iter < nbIterations; iter++) {
        let perte = 0;
        let gradW = 0;
        let gradB = 0;
        for (let i = 0; i < n; i++) {
            const erreur = w * zs[i] + b - ys[i];
            perte += erreur * erreur;
            gradW += erreur * zs[i];
            gradB += erreur;
        }
        perte /= n;
        gradW = (2 / n) * gradW; // ∂L/∂w
        gradB = (2 / n) * gradB; // ∂L/∂b

        pertes.push(perte);
        etapes.push(versOrigine(w, b));

        if (!Number.isFinite(perte)) break;

        w -= pas * gradW;
        b -= pas * gradB;
    }

    const premiere = pertes[0];
    const derniere = pertes[pertes.length - 1];
    const diverge = !Number.isFinite(derniere) || derniere > premiere;

    const finale = versOrigine(w, b);
    return { pente: finale.pente, ordonnee: finale.ordonnee, pertes, etapes, diverge };
}

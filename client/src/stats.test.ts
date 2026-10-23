/**
 * stats.test.ts : vérifie stats.ts avec un petit jeu de 5 points dont les
 * résultats ont été calculés à la main. Mêmes résultats = code fiable.
 */
import { describe, it, expect } from "vitest";
import {
    mean, median, sd, pearson, linreg, predireLineaire, mse, r2,
    creerAleatoire, genererSynthetique, decouperEntrainementTest,
    polyfit, predirePoly, gradientDescentLinear,
} from "./stats";

const xs = [1, 2, 3, 4, 5];
const ys = [3, 5, 6, 9, 11];

describe("Statistiques descriptives", () => {
    it("calcule la moyenne, la médiane et l'écart-type", () => {
        expect(mean(xs)).toBe(3);
        expect(mean(ys)).toBeCloseTo(6.8);
        expect(median(ys)).toBe(6);
        expect(median([4, 1, 3, 2])).toBe(2.5);
        expect(sd(xs)).toBeCloseTo(Math.sqrt(2.5), 6);
    });

    it("calcule la corrélation de Pearson", () => {
        expect(pearson(xs, ys)).toBeCloseTo(20 / Math.sqrt(408), 6); // ≈ 0,990
        expect(pearson(xs, xs.map((x) => -x))).toBeCloseTo(-1, 6);   // lien parfait décroissant
    });
});

describe("Régression linéaire et évaluation", () => {
    it("trouve la droite des moindres carrés", () => {
        const d = linreg(xs, ys);
        expect(d.pente).toBeCloseTo(2, 6);      // 20 / 10
        expect(d.ordonnee).toBeCloseTo(0.8, 6); // 6,8 − 2 × 3
        // Sans bruit, on retrouve exactement la règle cachée y = 2x + 1
        const parfait = linreg([0, 1, 2, 3], [1, 3, 5, 7]);
        expect(parfait.pente).toBeCloseTo(2, 6);
        expect(parfait.ordonnee).toBeCloseTo(1, 6);
    });

    it("évalue le modèle avec la MSE et le R²", () => {
        const d = linreg(xs, ys);
        const yPred = xs.map((x) => predireLineaire(d, x));
        expect(mse(ys, yPred)).toBeCloseTo(0.16, 6);                // 0,8 / 5
        expect(r2(ys, yPred)).toBeCloseTo(1 - 0.8 / 40.8, 6);       // ≈ 0,980
        expect(r2(ys, yPred)).toBeCloseTo(pearson(xs, ys) ** 2, 6); // propriété : R² = r²
        expect(r2(ys, ys.map(() => 0))).toBeLessThan(0);            // modèle nul : pire que la moyenne
    });
});

describe("Hasard reproductible et découpage", () => {
    it("génère des données reproductibles et les découpe en 70/30", () => {
        // 1) Même graine → exactement la même suite ; autre graine → autre suite
        const a = creerAleatoire(42);
        const b = creerAleatoire(42);
        const c = creerAleatoire(43);
        const va = a();
        expect(b()).toBe(va);
        expect(c()).not.toBe(va);

        // 2) Les nombres tombent toujours dans [0, 1)
        const r = creerAleatoire(1);
        for (let i = 0; i < 1000; i++) {
            const v = r();
            expect(v).toBeGreaterThanOrEqual(0);
            expect(v).toBeLessThan(1);
        }

        // 3) Avec beaucoup de points, la régression retrouve y = 2x + 1
        const grand = genererSynthetique(400, 1);
        const d = linreg(grand.map((p) => p.x), grand.map((p) => p.y));
        expect(Math.abs(d.pente - 2)).toBeLessThan(0.15);
        expect(Math.abs(d.ordonnee - 1)).toBeLessThan(0.8);

        // 4) Découpage de 40 points : 28 en entraînement, 12 en test, aucun perdu
        const pts = genererSynthetique(40, 1);
        const { entrainement, test } = decouperEntrainementTest(pts, 0.7, 7);
        expect(entrainement.length).toBe(28);
        expect(test.length).toBe(12);
        expect(new Set([...entrainement, ...test]).size).toBe(40);
    });
});

describe("Modèles avancés", () => {
    it("ajuste un polynôme, montre le surapprentissage et apprend par descente de gradient", () => {
        // --- Régression polynomiale ---

        // 1) Le degré 1 donne la même droite que linreg (2x + 0,8)
        const p1 = polyfit(xs, ys, 1);
        expect(predirePoly(p1, 0)).toBeCloseTo(0.8, 6);
        expect(predirePoly(p1, 3)).toBeCloseTo(6.8, 6);

        // 2) Sans bruit, le degré 2 retrouve exactement y = x² − 3x + 2
        const xq = [0, 1, 2, 3, 4, 5];
        const yq = xq.map((x) => x * x - 3 * x + 2);
        const p2 = polyfit(xq, yq, 2);
        expect(predirePoly(p2, 10)).toBeCloseTo(72, 4);

        // 3) Plus le degré monte, plus l'erreur d'entraînement baisse
        const { entrainement } = decouperEntrainementTest(genererSynthetique(40, 1), 0.7, 7);
        const ex = entrainement.map((p) => p.x);
        const ey = entrainement.map((p) => p.y);
        const erreurEntr = (d: number) => {
            const p = polyfit(ex, ey, d);
            return mse(ey, ex.map((x) => predirePoly(p, x)));
        };
        expect(erreurEntr(9)).toBeLessThanOrEqual(erreurEntr(1));

        // --- Descente de gradient ---

        // 4) Deux chemins vers la même droite : on retrouve la solution exacte
        const exacte = linreg(xs, ys);
        const gd = gradientDescentLinear(xs, ys, 0.1, 200);
        expect(gd.pente).toBeCloseTo(exacte.pente, 4);
        expect(gd.ordonnee).toBeCloseTo(exacte.ordonnee, 4);
        expect(gd.diverge).toBe(false);

        // 5) La perte descend : la dernière est bien plus petite que la première
        expect(gd.pertes[gd.pertes.length - 1]).toBeLessThan(gd.pertes[0]);

        // 6) Avec un pas de 1,1, l'apprentissage diverge
        expect(gradientDescentLinear(xs, ys, 1.1, 200).diverge).toBe(true);
    });
});

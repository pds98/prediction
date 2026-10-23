import { useEffect, useMemo, useState } from "react";
import "./App.css";
import {
  mean, median, sd, pearson, linreg, mse, r2,
  polyfit, predirePoly, genererSynthetique,
  decouperEntrainementTest, gradientDescentLinear,
} from "./stats";
import type { Point, Droite, ResultatDescente } from "./stats";

const API = import.meta.env.VITE_API_URL ?? "http://localhost:4006";
const GRAINE_DECOUPAGE = 7;
const MIN_POINTS = 6;
const NB_ITERATIONS = 100;

// Graphique principal (unités SVG)
const L = 600;
const H = 380;
const ZONE = { gauche: 55, droite: L - 20, haut: 20, bas: H - 45 };

// Courbe de perte
const LP = 300;
const HP = 110;

type Onglet = "synthetique" | "kiosque";
type EtatServeur = "chargement" | "ok" | "injoignable";

interface Analyse {
  entrainement: Point[];
  test: Point[];
  moyX: number; medX: number; sdX: number;
  moyY: number; medY: number; sdY: number;
  r: number;
  exacte: Droite;
  degreEffectif: number;
  courbe: (x: number) => number;
  erreurEntr: number;
  erreurTest: number;
  r2Test: number;
  surapprentissage: boolean;
}

/** Nombre au format fr-CA (virgule décimale). */
function fmt(v: number, decimales = 2): string {
  if (!Number.isFinite(v)) return "∞";
  return v.toLocaleString("fr-CA", {
    minimumFractionDigits: decimales,
    maximumFractionDigits: decimales,
  });
}

function interpreterR(r: number): string {
  const a = Math.abs(r);
  if (a < 0.2) return "lien linéaire très faible ou absent";
  const force = a >= 0.7 ? "fort" : a >= 0.4 ? "modéré" : "faible";
  return `lien linéaire ${force} et ${r > 0 ? "croissant" : "décroissant"}`;
}

/** Limite les coordonnées SVG quand une courbe s'emballe. */
const borne = (v: number) => Math.max(-2000, Math.min(2000, v));

const graduations = (min: number, max: number, n = 5) =>
  Array.from({ length: n + 1 }, (_, i) => min + (i * (max - min)) / n);

function calculerDomaine(points: Point[], onglet: Onglet) {
  if (onglet === "kiosque") return { xMin: 0, xMax: 24, yMin: 0, yMax: 100 };
  const ys = points.map((p) => p.y);
  return {
    xMin: 0,
    xMax: 10,
    yMin: Math.floor(Math.min(...ys, 0) - 2),
    yMax: Math.ceil(Math.max(...ys, 21) + 2),
  };
}

export default function App() {
  const [onglet, setOnglet] = useState<Onglet>("synthetique");
  const [graine, setGraine] = useState(1);
  const [nbPoints, setNbPoints] = useState(40);
  const [pointsKiosque, setPointsKiosque] = useState<Point[]>([]);
  const [etatServeur, setEtatServeur] = useState<EtatServeur>("chargement");
  const [degre, setDegre] = useState(1);
  const [pas, setPas] = useState(0.1);
  const [resultat, setResultat] = useState<ResultatDescente | null>(null);
  const [etape, setEtape] = useState(0);
  const [heures, setHeures] = useState("");
  const [batterie, setBatterie] = useState("");
  const [message, setMessage] = useState("");

  const points = useMemo<Point[]>(
    () => (onglet === "synthetique" ? genererSynthetique(nbPoints, graine) : pointsKiosque),
    [onglet, nbPoints, graine, pointsKiosque]
  );

  const domaine = useMemo(() => calculerDomaine(points, onglet), [points, onglet]);

  const analyse = useMemo<Analyse | string>(() => {
    if (points.length < MIN_POINTS) return `Il faut au moins ${MIN_POINTS} points`;

    const { entrainement, test } = decouperEntrainementTest(points, 0.7, GRAINE_DECOUPAGE);
    const xsE = entrainement.map((p) => p.x);
    const ysE = entrainement.map((p) => p.y);
    const xsT = test.map((p) => p.x);
    const ysT = test.map((p) => p.y);
    const xsTous = points.map((p) => p.x);
    const ysTous = points.map((p) => p.y);

    try {
      const exacte = linreg(xsE, ysE);

      // Un degré d exige au moins d + 1 valeurs de x distinctes.
      const nbXDistincts = new Set(xsE).size;
      const degreEffectif = Math.max(1, Math.min(degre, nbXDistincts - 1));
      const poly = polyfit(xsE, ysE, degreEffectif);
      const courbe = (x: number) => predirePoly(poly, x);

      const erreurEntr = mse(ysE, xsE.map(courbe));
      const erreurTest = mse(ysT, xsT.map(courbe));

      return {
        entrainement, test,
        moyX: mean(xsTous), medX: median(xsTous), sdX: sd(xsTous),
        moyY: mean(ysTous), medY: median(ysTous), sdY: sd(ysTous),
        r: pearson(xsTous, ysTous),
        exacte, degreEffectif, courbe, erreurEntr, erreurTest,
        r2Test: r2(ysT, xsT.map(courbe)),
        surapprentissage: degreEffectif >= 3 && erreurTest > 2 * erreurEntr,
      };
    } catch (e) {
      return e instanceof Error ? e.message : "Calcul impossible";
    }
  }, [points, degre]);

  useEffect(() => {
    let annule = false;
    fetch(`${API}/api/points`)
      .then((rep) => {
        if (!rep.ok) throw new Error("Réponse invalide");
        return rep.json();
      })
      .then((data: { heures: number; batterie: number }[]) => {
        if (annule) return;
        setPointsKiosque(data.map((p) => ({ x: p.heures, y: p.batterie })));
        setEtatServeur("ok");
      })
      .catch(() => {
        if (!annule) setEtatServeur("injoignable");
      });
    return () => {
      annule = true;
    };
  }, []);

  // Animation de l'entraînement : une étape toutes les 40 ms
  useEffect(() => {
    if (!resultat) return;
    const id = window.setInterval(() => {
      setEtape((e) => {
        if (e >= resultat.etapes.length - 1) {
          window.clearInterval(id);
          return e;
        }
        return e + 1;
      });
    }, 40);
    return () => window.clearInterval(id);
  }, [resultat]);

  function changerOnglet(o: Onglet) {
    setOnglet(o);
    setResultat(null);
    setMessage("");
  }

  function nouveauTirage() {
    setGraine((g) => g + 1);
    setResultat(null);
  }

  function changerPas(v: number) {
    setPas(v);
    setResultat(null);
  }

  function entrainer() {
    if (typeof analyse === "string") return;
    const xsE = analyse.entrainement.map((p) => p.x);
    const ysE = analyse.entrainement.map((p) => p.y);
    setResultat(gradientDescentLinear(xsE, ysE, pas, NB_ITERATIONS));
    setEtape(0);
  }

  async function ajouterPoint() {
    // Le serveur revalide aussi ; la virgule décimale est acceptée.
    const h = Number(heures.replace(",", "."));
    const b = Number(batterie.replace(",", "."));
    if (heures.trim() === "" || !Number.isFinite(h) || h < 0 || h > 24) {
      setMessage("Valeurs refusées : les heures doivent être entre 0 et 24.");
      return;
    }
    if (batterie.trim() === "" || !Number.isFinite(b) || b < 0 || b > 100) {
      setMessage("Valeurs refusées : la batterie doit être entre 0 et 100.");
      return;
    }

    try {
      const rep = await fetch(`${API}/api/points`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ heures: h, batterie: b }),
      });
      if (!rep.ok) {
        const err = await rep.json().catch(() => ({}));
        setMessage(err.erreur ?? "Valeurs refusées par le serveur.");
        return;
      }
      const nouveau: { heures: number; batterie: number } = await rep.json();
      setPointsKiosque((liste) => [...liste, { x: nouveau.heures, y: nouveau.batterie }]);
      setEtatServeur("ok");
      setHeures("");
      setBatterie("");
      setResultat(null);
      setMessage("Merci ! Votre point a été ajouté.");
    } catch {
      setEtatServeur("injoignable");
      setMessage("Serveur injoignable : le jeu synthétique reste disponible.");
    }
  }

  // Valeurs → pixels SVG (en SVG, y = 0 est en haut, d'où l'inversion dans sy)
  const sx = (x: number) =>
    ZONE.gauche + ((x - domaine.xMin) / (domaine.xMax - domaine.xMin)) * (ZONE.droite - ZONE.gauche);
  const sy = (y: number) =>
    ZONE.bas - ((y - domaine.yMin) / (domaine.yMax - domaine.yMin)) * (ZONE.bas - ZONE.haut);

  // Courbe du modèle : 200 segments
  let cheminModele = "";
  if (typeof analyse !== "string") {
    for (let i = 0; i <= 200; i++) {
      const x = domaine.xMin + (i / 200) * (domaine.xMax - domaine.xMin);
      const y = analyse.courbe(x);
      if (!Number.isFinite(y)) continue;
      cheminModele += `${cheminModele === "" ? "M" : "L"}${sx(x).toFixed(1)},${borne(sy(y)).toFixed(1)} `;
    }
  }

  const droiteGD: Droite | null = resultat ? resultat.etapes[etape] : null;
  const animFinie = resultat !== null && etape >= resultat.etapes.length - 1;
  const penteGD = resultat ? (animFinie ? resultat.pente : droiteGD!.pente) : NaN;

  let cheminPerte = "";
  if (resultat) {
    const finies = resultat.pertes.filter(Number.isFinite);
    const maxPerte = Math.max(...finies, 1e-9);
    const total = resultat.pertes.length;
    for (let i = 0; i <= etape && i < total; i++) {
      const p = resultat.pertes[i];
      if (!Number.isFinite(p)) break;
      const px = (i / Math.max(1, total - 1)) * (LP - 10) + 5;
      const py = HP - 5 - (p / maxPerte) * (HP - 10);
      cheminPerte += `${i === 0 ? "M" : "L"}${px.toFixed(1)},${py.toFixed(1)} `;
    }
  }

  const titreX = onglet === "kiosque" ? "Heures sur le téléphone hier" : "x";
  const titreY = onglet === "kiosque" ? "% de batterie restant" : "y";

  return (
    <div className="atelier">
      <header>
        <h1>L'Atelier de la prédiction</h1>
        <p>Comment une machine « apprend » à partir de données, sans magie ni boîte noire.</p>
      </header>

      <nav className="onglets">
        <button className={onglet === "synthetique" ? "actif" : ""} onClick={() => changerOnglet("synthetique")}>
          Jeu synthétique
        </button>
        <button className={onglet === "kiosque" ? "actif" : ""} onClick={() => changerOnglet("kiosque")}>
          Kiosque des visiteurs
        </button>
      </nav>

      <main className="grille">
        <section className="carte">
          <svg viewBox={`0 0 ${L} ${H}`} role="img" aria-label="Nuage de points et modèle">
            <defs>
              <clipPath id="zone">
                <rect x={ZONE.gauche} y={ZONE.haut} width={ZONE.droite - ZONE.gauche} height={ZONE.bas - ZONE.haut} />
              </clipPath>
            </defs>

            <line className="axe" x1={ZONE.gauche} y1={ZONE.bas} x2={ZONE.droite} y2={ZONE.bas} />
            <line className="axe" x1={ZONE.gauche} y1={ZONE.haut} x2={ZONE.gauche} y2={ZONE.bas} />
            {graduations(domaine.xMin, domaine.xMax).map((v) => (
              <text key={`gx${v}`} className="graduation" x={sx(v)} y={ZONE.bas + 16} textAnchor="middle">
                {fmt(v, 0)}
              </text>
            ))}
            {graduations(domaine.yMin, domaine.yMax).map((v) => (
              <text key={`gy${v}`} className="graduation" x={ZONE.gauche - 8} y={sy(v) + 4} textAnchor="end">
                {fmt(v, 0)}
              </text>
            ))}
            <text className="titre-axe" x={(ZONE.gauche + ZONE.droite) / 2} y={H - 6} textAnchor="middle">
              {titreX}
            </text>
            <text className="titre-axe" x={14} y={(ZONE.haut + ZONE.bas) / 2} textAnchor="middle"
                  transform={`rotate(-90 14 ${(ZONE.haut + ZONE.bas) / 2})`}>
              {titreY}
            </text>

            {typeof analyse === "string"
              ? points.map((p, i) => <circle key={`n${i}`} className="pt-neutre" cx={sx(p.x)} cy={sy(p.y)} r={5} />)
              : (
                <>
                  {analyse.entrainement.map((p, i) => (
                    <circle key={`e${i}`} className="pt-entr" cx={sx(p.x)} cy={sy(p.y)} r={5} />
                  ))}
                  {analyse.test.map((p, i) => (
                    <circle key={`t${i}`} className="pt-test" cx={sx(p.x)} cy={sy(p.y)} r={5} />
                  ))}
                </>
              )}

            <g clipPath="url(#zone)">
              {cheminModele && <path className="courbe-modele" d={cheminModele} />}
              {droiteGD && Number.isFinite(droiteGD.pente) && Number.isFinite(droiteGD.ordonnee) && (
                <line
                  className="droite-gd"
                  x1={sx(domaine.xMin)}
                  y1={borne(sy(droiteGD.pente * domaine.xMin + droiteGD.ordonnee))}
                  x2={sx(domaine.xMax)}
                  y2={borne(sy(droiteGD.pente * domaine.xMax + droiteGD.ordonnee))}
                />
              )}
            </g>
          </svg>

          <div className="legende">
            <span className="leg-entr">Entraînement (70 %)</span>
            <span className="leg-test">Test (30 %), jamais vu par le modèle</span>
            <span className="leg-modele">Modèle de degré {typeof analyse === "string" ? degre : analyse.degreEffectif}</span>
            <span className="leg-gd">Descente de gradient</span>
          </div>

          {typeof analyse === "string" && <p className="alerte">{analyse}</p>}
        </section>

        <aside className="carte">
          <div className="bloc">
            {onglet === "synthetique" ? (
              <>
                <h2>Jeu synthétique : y = 2x + 1 + bruit</h2>
                <button className="principal" onClick={nouveauTirage}>Nouveau tirage</button>{" "}
                <select value={nbPoints} onChange={(e) => { setNbPoints(Number(e.target.value)); setResultat(null); }}>
                  <option value={40}>40 points</option>
                  <option value={400}>400 points</option>
                </select>
                <p className="petit">Tirage n° {graine}</p>
              </>
            ) : (
              <>
                <h2>Ajoutez votre point (anonyme et facultatif)</h2>
                <form className="formulaire" onSubmit={(e) => { e.preventDefault(); void ajouterPoint(); }}>
                  <div>
                    <label htmlFor="heures">Heures sur le téléphone hier</label>
                    <input id="heures" type="number" step="0.5" min={0} max={24}
                           value={heures} onChange={(e) => setHeures(e.target.value)} />
                  </div>
                  <div>
                    <label htmlFor="batterie">% de batterie restant</label>
                    <input id="batterie" type="number" step="1" min={0} max={100}
                           value={batterie} onChange={(e) => setBatterie(e.target.value)} />
                  </div>
                  <button className="principal" type="submit">Ajouter</button>
                </form>
                {message && <p className="alerte info">{message}</p>}
                {etatServeur === "injoignable" && (
                  <p className="alerte erreur">Serveur injoignable : le jeu synthétique reste disponible.</p>
                )}
                <p className="petit">{pointsKiosque.length} point(s) enregistré(s)</p>
              </>
            )}
          </div>

          {typeof analyse !== "string" && (
            <>
              <div className="bloc">
                <h2>Décrire les données</h2>
                <table className="chiffres">
                  <thead>
                    <tr><th></th><th>Moyenne</th><th>Médiane</th><th>Écart-type</th></tr>
                  </thead>
                  <tbody>
                    <tr><td>{titreX === "x" ? "x" : "Heures"}</td><td>{fmt(analyse.moyX)}</td><td>{fmt(analyse.medX)}</td><td>{fmt(analyse.sdX)}</td></tr>
                    <tr><td>{titreY === "y" ? "y" : "Batterie"}</td><td>{fmt(analyse.moyY)}</td><td>{fmt(analyse.medY)}</td><td>{fmt(analyse.sdY)}</td></tr>
                  </tbody>
                </table>
                <p>Corrélation r = <strong>{fmt(analyse.r)}</strong> : {interpreterR(analyse.r)}.</p>
              </div>

              <div className="bloc">
                <h2>Complexité du modèle</h2>
                <label htmlFor="degre">Degré du polynôme : <strong>{degre}</strong></label>
                <input id="degre" type="range" min={1} max={9} step={1}
                       value={degre} onChange={(e) => setDegre(Number(e.target.value))} />
                {analyse.degreEffectif < degre && (
                  <p className="petit">Pas assez de valeurs différentes : degré limité à {analyse.degreEffectif}.</p>
                )}
                <table className="chiffres">
                  <tbody>
                    <tr><td>Erreur d'entraînement (MSE)</td><td>{fmt(analyse.erreurEntr)}</td></tr>
                    <tr><td>Erreur de test (MSE)</td><td>{fmt(analyse.erreurTest)}</td></tr>
                    <tr><td>R² sur le test</td><td>{fmt(analyse.r2Test)}</td></tr>
                  </tbody>
                </table>
                {analyse.surapprentissage && (
                  <p className="alerte">
                    ⚠️ Surapprentissage : le modèle colle aux points d'entraînement,
                    mais se trompe beaucoup plus sur les points de test. Il a appris
                    les exemples par cœur au lieu de comprendre la tendance.
                  </p>
                )}
              </div>

              <div className="bloc">
                <h2>Apprentissage pas à pas</h2>
                <label htmlFor="pas">Pas (taux d'apprentissage) : <strong>{fmt(pas)}</strong></label>
                <input id="pas" type="range" min={0.01} max={1.2} step={0.01}
                       value={pas} onChange={(e) => changerPas(Number(e.target.value))} />
                <button className="principal" onClick={entrainer}>Entraîner</button>

                <div className="deux-pentes">
                  <div>
                    <span className="petit">Pente exacte (formule)</span>
                    <strong>{fmt(analyse.exacte.pente, 3)}</strong>
                  </div>
                  <div>
                    <span className="petit">Pente apprise (gradient)</span>
                    <strong>{resultat ? fmt(penteGD, 3) : "—"}</strong>
                  </div>
                </div>

                {resultat && (
                  <>
                    <span className="petit">
                      Perte (erreur) à l'étape {etape + 1} / {resultat.pertes.length}
                    </span>
                    <svg viewBox={`0 0 ${LP} ${HP}`} role="img" aria-label="Courbe de perte">
                      <line className="axe" x1={5} y1={HP - 5} x2={LP - 5} y2={HP - 5} />
                      <path className="courbe-perte" d={cheminPerte} />
                    </svg>
                  </>
                )}
                {animFinie && resultat!.diverge && (
                  <p className="alerte erreur">
                    ❌ L'entraînement a échoué : le pas est trop grand. À chaque étape,
                    la machine saute par-dessus la bonne réponse et l'erreur augmente.
                  </p>
                )}
                {!resultat && <p className="petit">Cliquez sur « Entraîner » pour voir la droite rose apprendre.</p>}
              </div>
            </>
          )}
        </aside>
      </main>

      <footer>Corrélation n'est pas causalité : un lien statistique ne prouve pas qu'une chose en cause une autre.</footer>
    </div>
  );
}

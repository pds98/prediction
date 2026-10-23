/**
 * server/src/index.ts : l'API du kiosque.
 * Son seul rôle : stocker et valider les points des visiteurs.
 * Aucun calcul statistique ici, tout se fait dans le navigateur.
 *
 * Routes :
 *   GET    /api/health   le serveur et la base fonctionnent-ils ?
 *   GET    /api/points   la liste des points du kiosque
 *   POST   /api/points   ajoute un point (après validation)
 *   DELETE /api/points   efface tout (protégé par ADMIN_TOKEN)
 */
import express from "express";
import cors from "cors";
import pg from "pg";

// ---------------------------------------------------------------------
// Configuration, lue dans les variables d'environnement (docker-compose.yml)
// ---------------------------------------------------------------------
const PORT = Number(process.env.PORT ?? 4006);
const ADMIN_TOKEN = process.env.ADMIN_TOKEN ?? "changez-moi";

// Un « pool » garde quelques connexions ouvertes et les réutilise :
// beaucoup plus rapide que d'ouvrir une connexion à chaque requête.
const pool = new pg.Pool({ connectionString: process.env.DATABASE_URL });

const app = express();
app.use(cors());         // autorise le client (servi sur un autre port) à appeler l'API
app.use(express.json()); // transforme le corps JSON des requêtes en objet

// ---------------------------------------------------------------------
// Création de la table au démarrage (si elle n'existe pas déjà)
// ---------------------------------------------------------------------
// Les contraintes CHECK sont une deuxième ligne de défense : même si une
// valeur absurde passait la validation du code, la base la refuserait.
// On ne stocke que deux nombres : aucune donnée personnelle.
async function creerTable(): Promise<void> {
    await pool.query(`
        CREATE TABLE IF NOT EXISTS points (
            id        SERIAL PRIMARY KEY,
            heures    DOUBLE PRECISION NOT NULL CHECK (heures BETWEEN 0 AND 24),
            batterie  DOUBLE PRECISION NOT NULL CHECK (batterie BETWEEN 0 AND 100),
            cree_le   TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    `);
}

// ---------------------------------------------------------------------
// Validation : on ne fait jamais confiance au navigateur
// ---------------------------------------------------------------------
// N'importe qui peut appeler l'API sans passer par l'interface (avec curl,
// par exemple). Le serveur doit donc tout revérifier lui-même.
function estDansIntervalle(v: unknown, min: number, max: number): v is number {
    return typeof v === "number" && Number.isFinite(v) && v >= min && v <= max;
}

// ---------------------------------------------------------------------
// GET /api/health : vérifie que l'API et la base répondent
// ---------------------------------------------------------------------
app.get("/api/health", async (_req, res) => {
    try {
        await pool.query("SELECT 1");
        res.json({ ok: true, base: "ok" });
    } catch {
        res.status(503).json({ ok: false, base: "injoignable" });
    }
});

// ---------------------------------------------------------------------
// GET /api/points : renvoie tous les points
// ---------------------------------------------------------------------
app.get("/api/points", async (_req, res) => {
    try {
        const r = await pool.query("SELECT heures, batterie FROM points ORDER BY id");
        res.json(r.rows); // ex. [{ "heures": 3, "batterie": 64 }, ...]
    } catch (e) {
        console.error(e);
        res.status(500).json({ erreur: "Erreur de la base de données" });
    }
});

// ---------------------------------------------------------------------
// POST /api/points : ajoute un point après validation
// ---------------------------------------------------------------------
app.post("/api/points", async (req, res) => {
    const { heures, batterie } = req.body ?? {};

    if (!estDansIntervalle(heures, 0, 24)) {
        return res.status(400).json({ erreur: "Valeurs refusées : les heures doivent être entre 0 et 24." });
    }
    if (!estDansIntervalle(batterie, 0, 100)) {
        return res.status(400).json({ erreur: "Valeurs refusées : la batterie doit être entre 0 et 100." });
    }

    try {
        // $1 et $2 sont des paramètres : les valeurs voyagent séparément de la
        // requête SQL, ce qui empêche toute injection SQL.
        const r = await pool.query(
            "INSERT INTO points (heures, batterie) VALUES ($1, $2) RETURNING heures, batterie",
            [heures, batterie]
        );
        res.status(201).json(r.rows[0]); // 201 = « créé »
    } catch (e) {
        console.error(e);
        res.status(500).json({ erreur: "Erreur de la base de données" });
    }
});

// ---------------------------------------------------------------------
// DELETE /api/points : remet le jeu à zéro (réservé à l'équipe)
// ---------------------------------------------------------------------
app.delete("/api/points", async (req, res) => {
    if (req.header("x-admin-token") !== ADMIN_TOKEN) {
        return res.status(401).json({ erreur: "Jeton administrateur invalide" }); // 401 = non autorisé
    }
    try {
        const r = await pool.query("DELETE FROM points");
        res.json({ supprimes: r.rowCount });
    } catch (e) {
        console.error(e);
        res.status(500).json({ erreur: "Erreur de la base de données" });
    }
});

// ---------------------------------------------------------------------
// Démarrage : on crée la table, puis on écoute les requêtes
// ---------------------------------------------------------------------
creerTable()
    .then(() => {
        app.listen(PORT, () => console.log(`API prête sur http://localhost:${PORT}`));
    })
    .catch((e) => {
        console.error("Impossible de préparer la base :", e);
        process.exit(1); // Docker verra l'échec dans les journaux
    });

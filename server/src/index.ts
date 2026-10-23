/**
 * API du kiosque : stocke et valide les points des visiteurs.
 *   GET    /api/health
 *   GET    /api/points
 *   POST   /api/points
 *   DELETE /api/points   (en-tête x-admin-token requis)
 */
import express from "express";
import cors from "cors";
import pg from "pg";

const PORT = Number(process.env.PORT ?? 4006);
const ADMIN_TOKEN = process.env.ADMIN_TOKEN ?? "changez-moi";

const pool = new pg.Pool({ connectionString: process.env.DATABASE_URL });

const app = express();
app.use(cors());
app.use(express.json());

// Les CHECK protègent aussi la base si une valeur invalide passait la validation.
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

function estDansIntervalle(v: unknown, min: number, max: number): v is number {
    return typeof v === "number" && Number.isFinite(v) && v >= min && v <= max;
}

app.get("/api/health", async (_req, res) => {
    try {
        await pool.query("SELECT 1");
        res.json({ ok: true, base: "ok" });
    } catch {
        res.status(503).json({ ok: false, base: "injoignable" });
    }
});

app.get("/api/points", async (_req, res) => {
    try {
        const r = await pool.query("SELECT heures, batterie FROM points ORDER BY id");
        res.json(r.rows);
    } catch (e) {
        console.error(e);
        res.status(500).json({ erreur: "Erreur de la base de données" });
    }
});

app.post("/api/points", async (req, res) => {
    const { heures, batterie } = req.body ?? {};

    if (!estDansIntervalle(heures, 0, 24)) {
        return res.status(400).json({ erreur: "Valeurs refusées : les heures doivent être entre 0 et 24." });
    }
    if (!estDansIntervalle(batterie, 0, 100)) {
        return res.status(400).json({ erreur: "Valeurs refusées : la batterie doit être entre 0 et 100." });
    }

    try {
        // Requête paramétrée : protège contre l'injection SQL.
        const r = await pool.query(
            "INSERT INTO points (heures, batterie) VALUES ($1, $2) RETURNING heures, batterie",
            [heures, batterie]
        );
        res.status(201).json(r.rows[0]);
    } catch (e) {
        console.error(e);
        res.status(500).json({ erreur: "Erreur de la base de données" });
    }
});

app.delete("/api/points", async (req, res) => {
    if (req.header("x-admin-token") !== ADMIN_TOKEN) {
        return res.status(401).json({ erreur: "Jeton administrateur invalide" });
    }
    try {
        const r = await pool.query("DELETE FROM points");
        res.json({ supprimes: r.rowCount });
    } catch (e) {
        console.error(e);
        res.status(500).json({ erreur: "Erreur de la base de données" });
    }
});

creerTable()
    .then(() => {
        app.listen(PORT, () => console.log(`API prête sur http://localhost:${PORT}`));
    })
    .catch((e) => {
        console.error("Impossible de préparer la base :", e);
        process.exit(1);
    });

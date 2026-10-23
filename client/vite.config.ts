import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5176,       // l'application est servie sur http://localhost:5176
    strictPort: true, // si le port est pris, on s'arrête au lieu d'en choisir un autre
    host: true,       // écoute sur toutes les interfaces (nécessaire dans Docker)
  },
});

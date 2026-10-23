# Image de base : Node 22 sur un Linux léger (Debian « slim »)
FROM node:22-slim

# Dossier de travail dans le conteneur
WORKDIR /app

# 1) On copie d'abord seulement package.json, puis on installe les dépendances.
#    Docker garde cette étape en cache : tant que les dépendances ne changent
#    pas, il ne réinstalle pas tout à chaque modification du code.
COPY package*.json ./
RUN npm install

# 2) Puis on copie le reste du projet (node_modules est exclu par .dockerignore)
COPY . .

EXPOSE 4006

# « npm start » lance tsx, qui exécute directement le TypeScript (pas de compilation)
CMD ["npm", "start"]

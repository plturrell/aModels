FROM node:20-alpine

WORKDIR /app

COPY hana/package.json ./
RUN npm install --production || true

COPY hana/. .

ENV PORT=8083

CMD ["node", "app.js"]

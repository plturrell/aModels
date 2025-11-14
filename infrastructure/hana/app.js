import express from 'express';

const app = express();
const port = process.env.PORT || 8083;

app.use(express.json());

app.get('/healthz', (_req, res) => {
  res.json({ status: 'ok' });
});

app.post('/sql', (req, res) => {
  const { query, args } = req.body ?? {};
  res.json({
    query,
    args,
    rows: [],
    message: 'HANA stub service – no real database connected.'
  });
});

app.listen(port, () => {
  console.log(`HANA stub service listening on port ${port}`);
});

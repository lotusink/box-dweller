// vite.config.js
import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import getPort, { portNumbers } from 'get-port'
import { ProxyAgent, setGlobalDispatcher } from 'undici'

const ANTHROPIC_ENDPOINT = 'https://api.anthropic.com/v1/messages'
const ANTHROPIC_VERSION  = '2023-06-01'

function readJsonBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = []
    req.on('data', c => chunks.push(c))
    req.on('end', () => {
      try {
        const raw = Buffer.concat(chunks).toString('utf8')
        resolve(raw ? JSON.parse(raw) : {})
      } catch (e) { reject(e) }
    })
    req.on('error', reject)
  })
}

function sendJson(res, status, body) {
  res.statusCode = status
  res.setHeader('Content-Type', 'application/json; charset=utf-8')
  res.end(JSON.stringify(body))
}

// Route every outbound request in this process through the given proxy URL.
// Also mirrors the value to the four common proxy env vars so downstream
// tooling that reads process.env picks up the same setting.
function installGlobalProxy(proxyUrl) {
  if (!proxyUrl) return
  setGlobalDispatcher(new ProxyAgent(proxyUrl))
  process.env.HTTP_PROXY  = proxyUrl
  process.env.HTTPS_PROXY = proxyUrl
  process.env.http_proxy  = proxyUrl
  process.env.https_proxy = proxyUrl
}

// Dev-only middleware that terminates /api/chat, injects the Anthropic
// credentials on the server side, and relays the upstream response
// verbatim (status code and body).
function anthropicProxy(apiKey) {
  return {
    name: 'anthropic-dev-proxy',
    configureServer(server) {
      server.middlewares.use('/api/chat', async (req, res, next) => {
        if (req.method !== 'POST') { next(); return }
        if (!apiKey) {
          sendJson(res, 500, { error: { message: 'ANTHROPIC_API_KEY is not set. Add ANTHROPIC_API_KEY=<your key> to prototype/box-dweller/.env and restart the dev server.' } })
          return
        }
        let payload
        try { payload = await readJsonBody(req) }
        catch (e) { sendJson(res, 400, { error: { message: 'Invalid JSON body: ' + e.message } }); return }
        try {
          const upstream = await fetch(ANTHROPIC_ENDPOINT, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'x-api-key': apiKey,
              'anthropic-version': ANTHROPIC_VERSION,
            },
            body: JSON.stringify(payload),
          })
          const text = await upstream.text()
          res.statusCode = upstream.status
          res.setHeader('Content-Type', upstream.headers.get('content-type') || 'application/json; charset=utf-8')
          res.end(text)
        } catch (e) {
          sendJson(res, 502, { error: { message: 'Upstream fetch failed: ' + (e.message || String(e)) } })
        }
      })
    },
  }
}

export default defineConfig(async ({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const apiKey   = env.ANTHROPIC_API_KEY || ''
  const proxyUrl = env.PROXY || ''
  installGlobalProxy(proxyUrl)
  const port = await getPort({ port: portNumbers(1500, 3000) })

  return {
    plugins: [react(), anthropicProxy(apiKey)],
    server: {
      port,
      strictPort: true,
    },
  }
})

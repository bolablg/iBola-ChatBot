// iBola service worker (ASSESSMENT.md PART 8.3).
//
// Registered ONLY by the chat page on its own origin (never by embed.js on the
// host portfolio, and never inside the iframe embed). Strict rules:
//   - Never respondWith on /ask-agentic, /feedback, or any text/event-stream:
//     full passthrough, or SSE streaming breaks.
//   - Navigations are network-first (a cache-first index would strand old
//     bundles across Cloud Run revisions), with an offline fallback page.
//   - Static assets are network-first with a versioned cache fallback, so a new
//     revision is picked up without a hard refresh and the app still loads
//     offline.
// CACHE_VERSION is bumped with the VERSION file so activating a new worker
// purges the previous precache.

const CACHE_VERSION = 'ibola-v1.2.0';

const PRECACHE = [
  '/',
  '/static/style.css',
  '/static/script.js',
  '/static/logo.svg',
  '/static/quotes.json',
  '/static/posts.json',
  '/static/offline.html',
  '/static/fonts/inter-tight-latin.woff2',
  '/static/fonts/jetbrains-mono-latin.woff2',
];

// Dynamic endpoints that must never be served from cache or intercepted.
const BYPASS_PREFIXES = [
  '/ask-agentic',
  '/ask',
  '/chat',
  '/feedback',
  '/contact-alert',
  '/welcome',
  '/health',
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches
      .open(CACHE_VERSION)
      .then((cache) => cache.addAll(PRECACHE))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(keys.filter((k) => k !== CACHE_VERSION).map((k) => caches.delete(k)))
      )
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  const req = event.request;

  // Only same-origin GET is eligible; everything else is a full passthrough.
  if (req.method !== 'GET') return;
  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return;

  // Never intercept API / feedback / SSE: streaming and POSTs must hit network.
  if (BYPASS_PREFIXES.some((p) => url.pathname.startsWith(p))) return;
  if ((req.headers.get('accept') || '').includes('text/event-stream')) return;

  // Navigations: network-first, fall back to cached shell then offline page.
  if (req.mode === 'navigate') {
    event.respondWith(
      fetch(req).catch(() =>
        caches.match(req).then((cached) => cached || caches.match('/static/offline.html'))
      )
    );
    return;
  }

  // Static assets: network-first, refresh the versioned cache, fall back to it.
  event.respondWith(
    fetch(req)
      .then((res) => {
        if (res && res.status === 200 && res.type === 'basic') {
          const copy = res.clone();
          caches.open(CACHE_VERSION).then((cache) => cache.put(req, copy));
        }
        return res;
      })
      .catch(() => caches.match(req))
  );
});

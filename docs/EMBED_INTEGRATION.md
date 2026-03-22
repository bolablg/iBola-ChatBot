# iBola Chat Widget — Embed Integration Guide

Embed the iBola AI chatbot as a floating widget on any `*.bolablg.com` website.

## Quick Start

Add this single script tag before `</body>` on any page:

```html
<script src="https://chat.bolablg.com/static/embed.js" defer></script>
```

A blue chat bubble appears at the bottom-right. Clicking it opens the iMessage-style chatbot in a 400x600 popup.

## Configuration Options

Customize the widget using `data-*` attributes on the script tag:

| Attribute         | Default          | Description                          |
|-------------------|------------------|--------------------------------------|
| `data-position`   | `bottom-right`   | Widget position: `bottom-right` or `bottom-left` |
| `data-theme`      | `auto`           | Theme: `auto`, `light`, or `dark`    |
| `data-open`       | `false`          | Auto-open the chat on page load      |
| `data-accent`     | `#2563EB`        | Accent color for the trigger button  |

### Example with all options:

```html
<script
  src="https://chat.bolablg.com/static/embed.js"
  data-position="bottom-left"
  data-theme="dark"
  data-open="true"
  data-accent="#7C3AED"
  defer
></script>
```

## How It Works

1. The script injects a floating trigger button (blue circle with chat icon) at the configured corner
2. On first click, it lazy-loads `https://chat.bolablg.com/?embed=true` in a hidden iframe
3. The iframe renders the chatbot in **iMessage style** (compact bubbles, pill input, status indicator)
4. Subsequent clicks toggle the popup open/closed
5. The chatbot communicates with the backend via the iframe — no API keys or auth needed on the host page

## Two UI Modes

The chatbot has two distinct visual styles:

- **Standalone** (`https://chat.bolablg.com`) — ChatGPT-style full-page interface with suggested prompts, flat message rows with avatars, centered content column
- **Embedded** (`?embed=true`) — iMessage-style compact widget with chat bubbles (gray bot / blue user), status indicator, no avatars, pill-shaped input

The mode is automatically detected. No configuration needed.

## Domain Restrictions

For security, embedding is restricted to:

- `*.bolablg.com` (any subdomain)
- `bolablg.com` (root domain)
- `localhost` and `127.0.0.1` (development only)

Attempting to embed on other domains will log a warning and the widget will not load.

This restriction is enforced at two levels:
1. **Client-side** — `embed.js` checks `window.location.hostname`
2. **Server-side** — CSP `frame-ancestors` header only allows `bolablg.com` and its subdomains

## URLs

| URL | Purpose |
|-----|---------|
| `https://chat.bolablg.com` | Production — standalone chatbot (custom domain) |
| `https://bolablg.com/chat` | Alias — should redirect/proxy to `chat.bolablg.com` |
| `https://ibola-chatbot-1055950842890.us-central1.run.app` | Cloud Run service URL (direct) |
| `https://chat.bolablg.com/?embed=true` | Embedded iMessage mode (used by the widget iframe) |

### Setting up `bolablg.com/chat`

To make `https://bolablg.com/chat` serve the same chatbot as `chat.bolablg.com`, configure one of these on your main site:

**Option A — Client-side redirect (simplest):**
```html
<!-- On bolablg.com, at /chat route -->
<meta http-equiv="refresh" content="0;url=https://chat.bolablg.com">
```

**Option B — Nginx reverse proxy:**
```nginx
location /chat {
    proxy_pass https://chat.bolablg.com;
    proxy_set_header Host chat.bolablg.com;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
}
```

**Option C — Iframe embed on a dedicated page:**
```html
<!-- On bolablg.com/chat page -->
<iframe
  src="https://chat.bolablg.com"
  style="width:100%;height:100vh;border:none;"
  title="iBola AI Chatbot"
></iframe>
```

## Embedding on a Specific Page (Inline)

If you want the chatbot inline (not as a floating widget), embed it directly:

```html
<iframe
  src="https://chat.bolablg.com/?embed=true"
  style="width:400px;height:600px;border:none;border-radius:16px;"
  title="iBola AI Chatbot"
></iframe>
```

## Development & Testing

### Local testing

Start the server locally and use the test page:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Then open: `http://localhost:8000/static/test-embed.html`

This test page simulates the widget embedded on a website using `localhost`.

### Docker testing

```bash
docker compose up -d
```

Then open: `http://localhost:8080/static/test-embed.html`

## Responsive Behavior

- **Desktop** (>480px): 400x600 popup, positioned at the configured corner
- **Mobile** (<=480px): Near full-screen popup (`100vw - 20px` wide, `100vh - 80px` tall)

## Security

- The widget uses `loading="lazy"` on the iframe for performance
- No cookies or auth tokens are shared between the host page and the chatbot
- All communication happens within the iframe via standard HTTP requests
- CSP `frame-ancestors` prevents unauthorized embedding
- CORS is restricted to `*.bolablg.com` origins
- User input is validated server-side against XSS and injection patterns
- All DOM rendering uses safe methods (`textContent`, `createElement`) — no `innerHTML`

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| Widget doesn't appear | Domain not allowed | Must be on `*.bolablg.com` or localhost |
| Empty iframe after click | CSP blocking | Check browser console for `frame-ancestors` errors |
| Chat loads slowly | Cold start | Cloud Run containers may take 3-5s on first request |
| Styles look wrong | Cached CSS | Hard refresh (Cmd+Shift+R) to clear cache |

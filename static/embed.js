// ============================================================
// iBola Chat Widget — Embeddable on any website
//
// Usage: Add this to any page:
//   <script src="https://chat.bolablg.com/static/embed.js" defer></script>
//
// Options (via data attributes on the script tag):
//   data-position="bottom-right"  (bottom-right | bottom-left)
//   data-theme="auto"             (auto | light | dark)
//   data-open="false"             (auto-open on load)
//   data-accent="#2563EB"         (accent color)
// ============================================================

(function () {
    'use strict';

    const CHAT_URL = 'https://chat.bolablg.com/?embed=true';

    // Restrict embedding to *.bolablg.com domains only
    const hostname = window.location.hostname;
    if (!hostname.endsWith('bolablg.com') && hostname !== 'localhost' && hostname !== '127.0.0.1') {
        console.warn('iBola Chat: embedding is only allowed on *.bolablg.com domains.');
        return;
    }

    // Read config from script tag
    const scriptTag = document.currentScript || document.querySelector('script[src*="embed.js"]');
    const config = {
        position: (scriptTag && scriptTag.dataset.position) || 'bottom-right',
        theme: (scriptTag && scriptTag.dataset.theme) || 'auto',
        autoOpen: (scriptTag && scriptTag.dataset.open) === 'true',
        accent: (scriptTag && scriptTag.dataset.accent) || '#2563EB',
    };

    // Build widget DOM
    const wrapper = document.createElement('div');
    wrapper.id = 'ibola-chat-widget';

    // Styles
    const style = document.createElement('style');
    style.textContent = `
        #ibola-chat-widget {
            position: fixed;
            ${config.position === 'bottom-left' ? 'left: 20px;' : 'right: 20px;'}
            bottom: 20px;
            z-index: 99999;
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
        }

        #ibola-chat-widget .ibola-trigger {
            width: 56px;
            height: 56px;
            border-radius: 50%;
            background: ${config.accent};
            border: none;
            padding: 0;
            margin: 0;
            outline: none;
            -webkit-appearance: none;
            appearance: none;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 4px 16px rgba(0,0,0,0.16);
            transition: transform 200ms ease, box-shadow 200ms ease;
        }

        #ibola-chat-widget .ibola-trigger:focus-visible {
            outline: 2px solid ${config.accent};
            outline-offset: 2px;
        }

        #ibola-chat-widget .ibola-trigger:hover {
            transform: scale(1.08);
            box-shadow: 0 6px 24px rgba(0,0,0,0.2);
        }

        #ibola-chat-widget .ibola-trigger svg {
            width: 24px;
            height: 24px;
            fill: white;
            transition: transform 200ms ease;
        }

        #ibola-chat-widget .ibola-trigger.open svg { transform: rotate(90deg); }

        #ibola-chat-widget .ibola-frame-container {
            position: absolute;
            ${config.position === 'bottom-left' ? 'left: 0;' : 'right: 0;'}
            bottom: 68px;
            width: 400px;
            height: 600px;
            max-height: calc(100vh - 100px);
            max-width: calc(100vw - 40px);
            border-radius: 16px;
            overflow: hidden;
            box-shadow: 0 12px 48px rgba(0,0,0,0.15);
            border: 1px solid rgba(0,0,0,0.08);
            opacity: 0;
            transform: translateY(12px) scale(0.96);
            pointer-events: none;
            transition: opacity 250ms ease, transform 250ms ease;
        }

        #ibola-chat-widget .ibola-frame-container.visible {
            opacity: 1;
            transform: translateY(0) scale(1);
            pointer-events: auto;
        }

        #ibola-chat-widget .ibola-frame-container iframe {
            width: 100%;
            height: 100%;
            border: none;
        }

        @media (max-width: 480px) {
            #ibola-chat-widget .ibola-frame-container {
                width: calc(100vw - 20px);
                height: calc(100vh - 80px);
                bottom: 68px;
                right: -10px;
                border-radius: 12px;
            }
        }
    `;
    document.head.appendChild(style);

    // Trigger button
    const trigger = document.createElement('button');
    trigger.className = 'ibola-trigger';
    trigger.setAttribute('aria-label', 'Open chat');
    trigger.title = 'Chat with iBola';

    // Chat icon SVG (safe DOM)
    const NS = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(NS, 'svg');
    svg.setAttribute('viewBox', '0 0 24 24');
    const path = document.createElementNS(NS, 'path');
    path.setAttribute('d', 'M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z');
    svg.appendChild(path);
    trigger.appendChild(svg);

    // Frame container
    const frameContainer = document.createElement('div');
    frameContainer.className = 'ibola-frame-container';

    const iframe = document.createElement('iframe');
    iframe.title = 'iBola Chat';
    iframe.setAttribute('loading', 'lazy');
    frameContainer.appendChild(iframe);

    wrapper.appendChild(frameContainer);
    wrapper.appendChild(trigger);
    document.body.appendChild(wrapper);

    // Toggle
    let isOpen = false;
    const toggle = () => {
        isOpen = !isOpen;
        trigger.classList.toggle('open', isOpen);
        frameContainer.classList.toggle('visible', isOpen);
        trigger.setAttribute('aria-label', isOpen ? 'Close chat' : 'Open chat');

        // Lazy-load iframe on first open
        if (isOpen && !iframe.src) {
            iframe.src = CHAT_URL;
        }
    };

    trigger.addEventListener('click', toggle);

    // Auto-open
    if (config.autoOpen) {
        setTimeout(toggle, 1500);
    }
})();

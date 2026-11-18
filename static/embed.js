// ============================================================
// iBola Chat Widget, embeddable on any website
//
// Usage: Add this to any page:
//   <script src="https://chat.bolablg.com/static/embed.js" defer></script>
//
// Options (via data attributes on the script tag):
//   data-position="bottom-right"  (bottom-right | bottom-left)
//   data-theme="auto"             (auto | light | dark)
//   data-open="false"             (auto-open on load)
//   data-accent="#111111"         (launcher color; defaults to portfolio ink)
// ============================================================

(function () {
    'use strict';

    // Restrict embedding to *.bolablg.com domains only
    const hostname = window.location.hostname;
    const isLoopback = hostname === 'localhost' || hostname === '127.0.0.1';
    if (!hostname.endsWith('bolablg.com') && !isLoopback) {
        console.warn('iBola Chat: embedding is only allowed on *.bolablg.com domains.');
        return;
    }

    // Read config from script tag
    const scriptTag = document.currentScript || document.querySelector('script[src*="embed.js"]');
    const chatUrl = isLoopback && scriptTag?.src
        ? new URL('/?embed=true', scriptTag.src).href
        : 'https://chat.bolablg.com/?embed=true';
    const panelSide = scriptTag?.dataset.position === 'bottom-left' ? 'left' : 'right';
    const config = {
        theme: (scriptTag && scriptTag.dataset.theme) || 'auto',
        autoOpen: (scriptTag && scriptTag.dataset.open) === 'true',
        accent: (scriptTag && scriptTag.dataset.accent) || '#111111',
    };

    // Build widget DOM
    const wrapper = document.createElement('div');
    wrapper.id = 'ibola-chat-widget';

    // Styles
    const style = document.createElement('style');
    style.textContent = `
        #ibola-chat-widget {
            --ibola-gap: 12px;
            position: fixed;
            width: max-content;
            ${panelSide}: 20px;
            bottom: 20px;
            z-index: 99999;
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
            transform: translate(var(--ibola-viewport-shift, 0px), var(--ibola-viewport-lift, 0px));
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
            /* Paper icon on the ink launcher. data-accent is expected to be a
               dark color (portfolio ink by default) so the icon stays legible. */
            fill: #fafaf7;
        }

        #ibola-chat-widget .ibola-trigger.open svg {
            fill: none;
            stroke: #fafaf7;
            stroke-width: 2;
            stroke-linecap: round;
        }

        #ibola-chat-widget .ibola-frame-container {
            box-sizing: border-box;
            position: absolute;
            ${panelSide}: 0;
            bottom: calc(100% + var(--ibola-gap));
            width: 400px;
            height: 600px;
            max-height: calc(var(--ibola-space-above, 100vh - 80px) - var(--ibola-gap) - 12px - env(safe-area-inset-top, 0px));
            max-width: min(calc(100vw - 40px), calc(var(--ibola-space-across, 100vw) - 12px));
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 12px 48px rgba(0,0,0,0.15);
            border: 1px solid rgba(0,0,0,0.08);
            visibility: hidden;
            opacity: 0;
            transform: translateY(12px) scale(0.96);
            pointer-events: none;
            transition: opacity 250ms ease, transform 250ms ease, visibility 0s linear 250ms;
        }

        #ibola-chat-widget .ibola-frame-container.visible {
            visibility: visible;
            opacity: 1;
            transform: translateY(0) scale(1);
            pointer-events: auto;
            transition-delay: 0s;
        }

        #ibola-chat-widget .ibola-frame-container iframe {
            display: block;
            width: 100%;
            height: 100%;
            border: none;
        }

        @media (max-width: 767px), (hover: none) and (max-height: 500px) {
            #ibola-chat-widget {
                --ibola-gap: 8px;
                ${panelSide}: max(12px, env(safe-area-inset-${panelSide}, 0px));
                bottom: max(12px, env(safe-area-inset-bottom, 0px));
            }

            #ibola-chat-widget .ibola-trigger {
                width: 44px;
                height: 44px;
            }

            #ibola-chat-widget .ibola-frame-container {
                width: 360px;
                height: max(0px, calc(var(--ibola-viewport-height, 100vh) * 0.75 - var(--ibola-launcher-height, 44px) - var(--ibola-gap)));
                border-radius: 12px;
            }
        }

        @media (prefers-reduced-motion: reduce) {
            #ibola-chat-widget .ibola-trigger,
            #ibola-chat-widget .ibola-frame-container { transition: none; }
        }
    `;
    document.head.appendChild(style);

    // Trigger button
    const trigger = document.createElement('button');
    trigger.className = 'ibola-trigger';
    trigger.type = 'button';
    trigger.setAttribute('aria-label', 'Open chat');
    trigger.setAttribute('aria-expanded', 'false');
    trigger.setAttribute('aria-controls', 'ibola-chat-panel');
    trigger.title = 'Chat with iBola';

    // Chat icon SVG (safe DOM)
    const NS = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(NS, 'svg');
    svg.setAttribute('viewBox', '0 0 24 24');
    svg.setAttribute('aria-hidden', 'true');
    const path = document.createElementNS(NS, 'path');
    const chatIcon = 'M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z';
    path.setAttribute('d', chatIcon);
    svg.appendChild(path);
    trigger.appendChild(svg);

    // Frame container
    const frameContainer = document.createElement('div');
    frameContainer.id = 'ibola-chat-panel';
    frameContainer.className = 'ibola-frame-container';
    frameContainer.inert = true;

    const iframe = document.createElement('iframe');
    iframe.title = 'iBola Chat';
    iframe.setAttribute('loading', 'lazy');
    frameContainer.appendChild(iframe);

    wrapper.appendChild(trigger);
    wrapper.appendChild(frameContainer);
    document.body.appendChild(wrapper);

    // Toggle
    let isOpen = false;
    let viewportLift = 0;
    let viewportShift = 0;
    const viewport = window.visualViewport;
    const fitPanel = () => {
        if (!isOpen) return;

        // Measure the host's launcher offset, then fit in the visible viewport.
        const { top, bottom, left, right } = wrapper.getBoundingClientRect();
        const visibleTop = viewport?.offsetTop ?? 0;
        const visibleBottom = visibleTop + (viewport?.height ?? window.innerHeight);
        const visibleLeft = viewport?.offsetLeft ?? 0;
        const visibleRight = visibleLeft + (viewport?.width ?? window.innerWidth);
        // The mobile height budget includes the launcher and its gap.
        wrapper.style.setProperty('--ibola-viewport-height', `${visibleBottom - visibleTop}px`);
        wrapper.style.setProperty('--ibola-launcher-height', `${bottom - top}px`);

        const layoutLeft = left - viewportShift;
        const layoutRight = right - viewportShift;
        const minShift = visibleLeft + 12 - layoutLeft;
        const maxShift = visibleRight - 12 - layoutRight;
        viewportShift = Math.min(Math.max(0, minShift), maxShift);
        const spaceAcross = panelSide === 'left'
            ? visibleRight - (layoutLeft + viewportShift)
            : layoutRight + viewportShift - visibleLeft;
        wrapper.style.setProperty('--ibola-viewport-shift', `${viewportShift}px`);
        wrapper.style.setProperty('--ibola-space-across', `${Math.max(0, spaceAcross)}px`);

        const layoutTop = top + viewportLift;
        viewportLift = Math.max(0, Math.ceil(bottom + viewportLift - visibleBottom + 12));
        wrapper.style.setProperty('--ibola-viewport-lift', `${-viewportLift}px`);
        const spaceAbove = Math.max(0, Math.floor(layoutTop - viewportLift - visibleTop));
        wrapper.style.setProperty('--ibola-space-above', `${spaceAbove}px`);
    };
    const trackEmbed = (name, params) => {
        if (typeof gtag === 'function') gtag('event', name, params);
    };
    const toggle = () => {
        isOpen = !isOpen;
        if (isOpen) {
            fitPanel();
        } else {
            viewportLift = 0;
            viewportShift = 0;
            wrapper.style.removeProperty('--ibola-viewport-lift');
            wrapper.style.removeProperty('--ibola-viewport-shift');
            trigger.focus({ preventScroll: true });
        }
        trigger.classList.toggle('open', isOpen);
        frameContainer.classList.toggle('visible', isOpen);
        frameContainer.inert = !isOpen;
        path.setAttribute('d', isOpen ? 'M6 6l12 12M18 6L6 18' : chatIcon);
        trigger.setAttribute('aria-label', isOpen ? 'Close chat' : 'Open chat');
        trigger.setAttribute('aria-expanded', String(isOpen));
        trackEmbed(isOpen ? 'chat_widget_opened' : 'chat_widget_closed', {
            page_url: window.location.href,
        });

        // Lazy-load iframe on first open
        if (isOpen && !iframe.src) {
            iframe.src = chatUrl;
        }
    };

    trigger.addEventListener('click', toggle);
    window.addEventListener('resize', fitPanel);
    viewport?.addEventListener('resize', fitPanel);
    viewport?.addEventListener('scroll', fitPanel);

    // Auto-open
    if (config.autoOpen) {
        setTimeout(toggle, 1500);
    }
})();

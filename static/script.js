// ============================================================
// iBola Chatbot — Frontend Application
// Vanilla JS, SSE streaming, feedback, dark mode, embeddable
// ============================================================

document.addEventListener('DOMContentLoaded', () => {
    // --- DOM ---
    const chatMessages = document.getElementById('chat-messages');
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const typingIndicator = document.getElementById('typing-indicator');
    const sendBtn = chatForm.querySelector('.send-btn');
    const themeToggle = document.getElementById('theme-toggle');
    const chatContainer = document.getElementById('chat-container');
    const modalOverlay = document.getElementById('modal-overlay');
    const modalIframe = document.getElementById('modal-iframe');
    const modalClose = document.getElementById('modal-close');

    // --- State ---
    const sessionId = `web-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
    let userLanguage = 'en';
    let chatEnded = false;
    let msgIndex = 0;

    // --- Theme ---
    const applyTheme = (theme) => {
        const isDark = theme === 'dark';
        document.body.classList.toggle('dark-mode', isDark);
        if (themeToggle) {
            themeToggle.querySelector('.icon-sun').style.display = isDark ? 'block' : 'none';
            themeToggle.querySelector('.icon-moon').style.display = isDark ? 'none' : 'block';
        }
        localStorage.setItem('theme', theme);
    };

    if (themeToggle) {
        themeToggle.addEventListener('click', () => {
            applyTheme(document.body.classList.contains('dark-mode') ? 'light' : 'dark');
        });
    }

    applyTheme(localStorage.getItem('theme') || 'light');

    // --- Helpers ---
    const scrollToBottom = () => {
        requestAnimationFrame(() => { chatMessages.scrollTop = chatMessages.scrollHeight; });
    };

    const showTyping = () => { typingIndicator.classList.add('visible'); scrollToBottom(); };
    const hideTyping = () => { typingIndicator.classList.remove('visible'); };

    // --- SVG icon builders (safe DOM, no innerHTML) ---
    const createSvg = (paths, size = 14) => {
        const NS = 'http://www.w3.org/2000/svg';
        const svg = document.createElementNS(NS, 'svg');
        svg.setAttribute('width', size);
        svg.setAttribute('height', size);
        svg.setAttribute('viewBox', '0 0 24 24');
        svg.setAttribute('fill', 'none');
        svg.setAttribute('stroke', 'currentColor');
        svg.setAttribute('stroke-width', '2');
        svg.setAttribute('stroke-linecap', 'round');
        svg.setAttribute('stroke-linejoin', 'round');
        svg.setAttribute('aria-hidden', 'true');
        paths.forEach(d => {
            const path = document.createElementNS(NS, 'path');
            path.setAttribute('d', d);
            svg.appendChild(path);
        });
        return svg;
    };

    const thumbUpIcon = () => createSvg([
        'M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3H14z',
        'M7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3',
    ]);

    const thumbDownIcon = () => createSvg([
        'M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3H10z',
        'M17 2h3a2 2 0 0 1 2 2v7a2 2 0 0 1-2 2h-3',
    ]);

    // --- Build feedback row (safe DOM) ---
    const createFeedbackRow = (idx) => {
        const fb = document.createElement('div');
        fb.classList.add('msg-feedback');

        const upBtn = document.createElement('button');
        upBtn.classList.add('feedback-btn');
        upBtn.dataset.score = '1';
        upBtn.dataset.idx = idx;
        upBtn.setAttribute('aria-label', 'Good response');
        upBtn.title = 'Good response';
        upBtn.appendChild(thumbUpIcon());

        const downBtn = document.createElement('button');
        downBtn.classList.add('feedback-btn');
        downBtn.dataset.score = '0';
        downBtn.dataset.idx = idx;
        downBtn.setAttribute('aria-label', 'Bad response');
        downBtn.title = 'Bad response';
        downBtn.appendChild(thumbDownIcon());

        [upBtn, downBtn].forEach(btn => {
            btn.addEventListener('click', () => handleFeedback(btn, fb));
        });

        fb.appendChild(upBtn);
        fb.appendChild(downBtn);
        return fb;
    };

    // --- Message Rendering ---
    const addMessage = (text, sender, opts = {}) => {
        const row = document.createElement('div');
        row.classList.add('msg-row', `msg-row--${sender}`);

        const bubble = document.createElement('div');
        bubble.classList.add('msg', `msg--${sender}`);
        bubble.textContent = text;
        row.appendChild(bubble);

        if (sender === 'bot' && !opts.noFeedback) {
            row.appendChild(createFeedbackRow(msgIndex++));
        }

        chatMessages.insertBefore(row, typingIndicator);
        scrollToBottom();
        return bubble;
    };

    // --- Typing Animation ---
    const typeMessage = (text, sender = 'bot', opts = {}) => {
        return new Promise(resolve => {
            showTyping();
            const row = document.createElement('div');
            row.classList.add('msg-row', `msg-row--${sender}`);

            const bubble = document.createElement('div');
            bubble.classList.add('msg', `msg--${sender}`);
            row.appendChild(bubble);

            chatMessages.insertBefore(row, typingIndicator);
            hideTyping();

            let i = 0;
            const speed = opts.speed || 25;
            const tick = () => {
                bubble.textContent += text.charAt(i);
                i++;
                scrollToBottom();
                if (i < text.length) {
                    setTimeout(tick, speed);
                } else {
                    if (sender === 'bot' && !opts.noFeedback) {
                        row.appendChild(createFeedbackRow(msgIndex++));
                    }
                    resolve();
                }
            };
            tick();
        });
    };

    // --- Feedback ---
    const handleFeedback = async (btn, container) => {
        const score = parseFloat(btn.dataset.score);
        const idx = parseInt(btn.dataset.idx);

        container.querySelectorAll('.feedback-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');

        setTimeout(() => {
            while (container.firstChild) container.removeChild(container.firstChild);
            const thanks = document.createElement('span');
            thanks.classList.add('feedback-thanks');
            thanks.textContent = 'Thanks for your feedback';
            container.appendChild(thanks);
        }, 600);

        try {
            await fetch('/feedback', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: sessionId, message_index: idx, score }),
            });
        } catch (e) { /* silent */ }
    };

    // --- Action Buttons ---
    const addActionButtons = (actions) => {
        const group = document.createElement('div');
        group.classList.add('action-group');

        actions.forEach(action => {
            const btn = document.createElement('button');
            btn.classList.add('action-btn');
            if (action.primary) btn.classList.add('action-btn--primary');
            btn.textContent = action.text;
            btn.title = action.description || '';

            btn.addEventListener('click', () => {
                if (action.type === 'contact_booking') {
                    window.open(action.url, 'booking', 'width=600,height=700,scrollbars=yes');
                    sendContactAlert('booking_request', action.session_id, action.chat_history);
                } else if (action.type === 'contact_email') {
                    window.open(action.url, '_blank');
                    sendContactAlert('email_request', action.session_id, action.chat_history);
                }
            });

            group.appendChild(btn);
        });

        chatMessages.insertBefore(group, typingIndicator);
        scrollToBottom();
    };

    // --- Contact Alert ---
    const sendContactAlert = async (type, sid, history) => {
        try {
            await fetch('/contact-alert', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    contact_type: type,
                    session_id: sid || sessionId,
                    chat_history: history || [],
                    timestamp: new Date().toISOString(),
                }),
            });
        } catch (e) { /* silent */ }
    };

    // --- End Chat ---
    const endChat = () => {
        chatEnded = true;
        chatContainer.classList.add('chat-ended');
        userInput.disabled = true;
        userInput.placeholder = 'Chat ended';
        sendBtn.disabled = true;
    };

    // --- Form Submit ---
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const text = userInput.value.trim();
        if (!text || chatEnded) return;

        addMessage(text, 'user');
        userInput.value = '';
        showTyping();

        try {
            const res = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    user_input: text,
                    session_id: sessionId,
                    user_language: userLanguage,
                }),
            });

            hideTyping();

            if (res.ok) {
                const data = await res.json();
                if (data.answer) await typeMessage(data.answer, 'bot');
                if (data.actions && data.actions.length > 0) addActionButtons(data.actions);
                if (data.should_end_chat && !chatEnded) endChat();
            } else {
                addMessage('Something went wrong. Please try again.', 'bot', { noFeedback: true });
            }
        } catch (err) {
            hideTyping();
            addMessage('Connection error. Please check your internet and try again.', 'bot', { noFeedback: true });
        }
    });

    // --- Modal ---
    if (modalClose) {
        modalClose.addEventListener('click', () => {
            modalOverlay.classList.remove('visible');
            modalIframe.src = '';
        });
    }
    if (modalOverlay) {
        modalOverlay.addEventListener('click', (e) => {
            if (e.target === modalOverlay) {
                modalOverlay.classList.remove('visible');
                modalIframe.src = '';
            }
        });
    }

    // --- Placeholder per language ---
    const placeholders = {
        en: "Ask about Bolaji\u2019s experience\u2026",
        fr: "Posez une question sur Bolaji\u2026",
        es: "\u00bfPreguntas sobre Bolaji?",
        de: "Fragen \u00fcber Bolaji\u2026",
        pt: "Pergunte sobre Bolaji\u2026",
        it: "Chiedi di Bolaji\u2026",
        ru: "\u0421\u043f\u0440\u043e\u0441\u0438\u0442\u0435 \u043e Bolaji\u2026",
        zh: "\u95ee\u5173\u4e8eBolaji\u7684\u95ee\u9898\u2026",
        ja: "Bolaji\u306b\u3064\u3044\u3066\u8cea\u554f\u2026",
        ko: "Bolaji\uc5d0 \ub300\ud574 \ubb3c\uc5b4\ubcf4\uc138\uc694\u2026",
    };

    // --- Init ---
    const init = async () => {
        const browserLang = navigator.language || 'en';

        try {
            const res = await fetch('/welcome', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: sessionId, browser_language: browserLang }),
            });

            if (res.ok) {
                const data = await res.json();
                userLanguage = data.detected_language || 'en';
                const msgs = data.welcome_messages || [];
                userInput.placeholder = placeholders[userLanguage] || placeholders.en;

                if (msgs.length > 0) {
                    await new Promise(r => setTimeout(r, 400));
                    await typeMessage(msgs[0], 'bot', { noFeedback: true });
                }
            } else {
                throw new Error('Welcome failed');
            }
        } catch (e) {
            userInput.placeholder = placeholders.en;
            await new Promise(r => setTimeout(r, 400));
            await typeMessage(
                "Hello! I\u2019m iBola, Bolaji\u2019s AI assistant. Ask me about his experience, education, or skills.",
                'bot', { noFeedback: true }
            );
        }

        userInput.focus();
    };

    init();
});

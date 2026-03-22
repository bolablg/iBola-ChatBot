// ============================================================
// iBola Chatbot — ChatGPT-Style Frontend
// Vanilla JS, markdown rendering, suggestions, SSE, feedback
// ============================================================

document.addEventListener('DOMContentLoaded', () => {
    // --- DOM ---
    const chatMain = document.getElementById('chat-main');
    const chatMessages = document.getElementById('chat-messages');
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const typingIndicator = document.getElementById('typing-indicator');
    const sendBtn = chatForm.querySelector('.send-btn');
    const themeToggle = document.getElementById('theme-toggle');
    const chatApp = document.getElementById('chat-app');
    const welcomeScreen = document.getElementById('welcome-screen');
    const suggestionsGrid = document.getElementById('suggestions-grid');
    const modalOverlay = document.getElementById('modal-overlay');
    const modalIframe = document.getElementById('modal-iframe');
    const modalClose = document.getElementById('modal-close');

    // --- State ---
    const sessionId = `web-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
    const isEmbedMode = document.documentElement.classList.contains('minimal');
    let userLanguage = 'en';
    let chatEnded = false;
    let msgIndex = 0;
    let hasUserSent = false;

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
        requestAnimationFrame(() => { chatMain.scrollTop = chatMain.scrollHeight; });
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

    const userIcon = () => createSvg([
        'M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2',
        'M12 3a4 4 0 1 0 0 8 4 4 0 0 0 0-8z',
    ], 16);

    // --- Safe Markdown Renderer (DOM-based, no innerHTML) ---
    const appendInlineMarkdown = (parent, text) => {
        const pattern = /(\*\*(.+?)\*\*|\*(.+?)\*|`(.+?)`|\[(.+?)\]\((.+?)\))/g;
        let lastIndex = 0;
        let match;

        while ((match = pattern.exec(text)) !== null) {
            if (match.index > lastIndex) {
                parent.appendChild(document.createTextNode(text.slice(lastIndex, match.index)));
            }

            if (match[2]) {
                const strong = document.createElement('strong');
                strong.textContent = match[2];
                parent.appendChild(strong);
            } else if (match[3]) {
                const em = document.createElement('em');
                em.textContent = match[3];
                parent.appendChild(em);
            } else if (match[4]) {
                const code = document.createElement('code');
                code.textContent = match[4];
                parent.appendChild(code);
            } else if (match[5] && match[6]) {
                const a = document.createElement('a');
                a.textContent = match[5];
                a.href = match[6];
                a.target = '_blank';
                a.rel = 'noopener noreferrer';
                parent.appendChild(a);
            }

            lastIndex = match.index + match[0].length;
        }

        if (lastIndex < text.length) {
            parent.appendChild(document.createTextNode(text.slice(lastIndex)));
        }
    };

    const renderMarkdown = (text) => {
        const container = document.createElement('div');
        const blocks = text.split(/\n{2,}/);

        blocks.forEach(block => {
            block = block.trim();
            if (!block) return;

            // Code block
            if (block.startsWith('```')) {
                const pre = document.createElement('pre');
                const code = document.createElement('code');
                code.textContent = block.replace(/^```\w*\n?/, '').replace(/\n?```$/, '');
                pre.appendChild(code);
                container.appendChild(pre);
                return;
            }

            // Unordered list
            if (/^[-*] /.test(block)) {
                const ul = document.createElement('ul');
                block.split('\n').forEach(line => {
                    if (/^[-*]\s+/.test(line)) {
                        const li = document.createElement('li');
                        appendInlineMarkdown(li, line.replace(/^[-*]\s+/, ''));
                        ul.appendChild(li);
                    }
                });
                container.appendChild(ul);
                return;
            }

            // Ordered list
            if (/^\d+\.\s/.test(block)) {
                const ol = document.createElement('ol');
                block.split('\n').forEach(line => {
                    if (/^\d+\.\s+/.test(line)) {
                        const li = document.createElement('li');
                        appendInlineMarkdown(li, line.replace(/^\d+\.\s+/, ''));
                        ol.appendChild(li);
                    }
                });
                container.appendChild(ol);
                return;
            }

            // Paragraph (handle single newlines within block)
            block.split('\n').forEach(line => {
                const p = document.createElement('p');
                appendInlineMarkdown(p, line);
                container.appendChild(p);
            });
        });

        return container;
    };

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

    // --- Create message row with avatar ---
    const createMessageRow = (sender) => {
        const row = document.createElement('div');
        row.classList.add('msg-row', `msg-row--${sender}`);

        const avatar = document.createElement('div');
        avatar.classList.add('msg-avatar');

        if (sender === 'bot') {
            const img = document.createElement('img');
            img.src = 'https://files.bolablg.com/images/ji_fav_192.png';
            img.alt = 'iBola';
            img.width = 28;
            img.height = 28;
            avatar.appendChild(img);
        } else {
            avatar.classList.add('msg-avatar--user');
            avatar.appendChild(userIcon());
        }
        row.appendChild(avatar);

        const content = document.createElement('div');
        content.classList.add('msg-content');
        row.appendChild(content);

        return { row, content };
    };

    // --- Hide welcome screen ---
    const hideWelcomeScreen = () => {
        if (welcomeScreen) {
            welcomeScreen.style.display = 'none';
        }
    };

    // --- Message Rendering ---
    const addMessage = (text, sender, opts = {}) => {
        if (sender === 'user' && !hasUserSent) {
            hasUserSent = true;
            hideWelcomeScreen();
        }

        const { row, content } = createMessageRow(sender);

        if (sender === 'bot' && !opts.plain) {
            content.appendChild(renderMarkdown(text));
        } else {
            content.textContent = text;
        }

        if (sender === 'bot' && !opts.noFeedback) {
            content.appendChild(createFeedbackRow(msgIndex++));
        }

        chatMessages.insertBefore(row, typingIndicator);
        scrollToBottom();
        return content;
    };

    // --- Typing Animation ---
    const typeMessage = (text, sender = 'bot', opts = {}) => {
        return new Promise(resolve => {
            if (sender === 'user' && !hasUserSent) {
                hasUserSent = true;
                hideWelcomeScreen();
            }

            showTyping();

            const { row, content } = createMessageRow(sender);
            content.classList.add('msg-content--plain', 'msg-content--streaming');

            chatMessages.insertBefore(row, typingIndicator);
            hideTyping();

            let i = 0;
            const speed = opts.speed || 20;
            const tick = () => {
                content.textContent = text.slice(0, ++i);
                scrollToBottom();
                if (i < text.length) {
                    setTimeout(tick, speed);
                } else {
                    // Replace with rendered markdown
                    content.classList.remove('msg-content--plain', 'msg-content--streaming');
                    content.textContent = '';
                    content.appendChild(renderMarkdown(text));

                    if (sender === 'bot' && !opts.noFeedback) {
                        content.appendChild(createFeedbackRow(msgIndex++));
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
        chatApp.classList.add('chat-ended');
        userInput.disabled = true;
        userInput.placeholder = 'Chat ended';
        sendBtn.disabled = true;
    };

    // --- Textarea Auto-Grow ---
    userInput.addEventListener('input', () => {
        userInput.style.height = 'auto';
        userInput.style.height = Math.min(userInput.scrollHeight, 200) + 'px';
    });

    userInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            chatForm.dispatchEvent(new Event('submit', { cancelable: true }));
        }
    });

    // --- Form Submit ---
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const text = userInput.value.trim();
        if (!text || chatEnded) return;

        addMessage(text, 'user', { plain: true });
        userInput.value = '';
        userInput.style.height = 'auto';
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

    // --- Suggestion Cards ---
    const suggestions = {
        en: [
            "What are Bolaji's key skills?",
            "Tell me about his work experience",
            "What is his educational background?",
            "How can I contact Bolaji?",
        ],
        fr: [
            "Quelles sont les comp\u00e9tences de Bolaji ?",
            "Parlez-moi de son exp\u00e9rience professionnelle",
            "Quelle est sa formation ?",
            "Comment contacter Bolaji ?",
        ],
        es: [
            "\u00bfCu\u00e1les son las habilidades de Bolaji?",
            "Cu\u00e9ntame sobre su experiencia laboral",
            "\u00bfCu\u00e1l es su formaci\u00f3n acad\u00e9mica?",
            "\u00bfC\u00f3mo puedo contactar a Bolaji?",
        ],
        de: [
            "Was sind Bolajis wichtigste F\u00e4higkeiten?",
            "Erz\u00e4hlen Sie mir von seiner Berufserfahrung",
            "Welche Ausbildung hat er?",
            "Wie kann ich Bolaji kontaktieren?",
        ],
        pt: [
            "Quais s\u00e3o as habilidades de Bolaji?",
            "Fale sobre sua experi\u00eancia profissional",
            "Qual \u00e9 sua forma\u00e7\u00e3o?",
            "Como posso contactar Bolaji?",
        ],
    };

    const welcomeTexts = {
        en: { title: "How can I help you?", subtitle: "Ask me anything about Bolaji's professional life" },
        fr: { title: "Comment puis-je vous aider ?", subtitle: "Posez-moi une question sur la vie professionnelle de Bolaji" },
        es: { title: "\u00bfC\u00f3mo puedo ayudarte?", subtitle: "Preg\u00fantame sobre la vida profesional de Bolaji" },
        de: { title: "Wie kann ich Ihnen helfen?", subtitle: "Fragen Sie mich \u00fcber Bolajis Berufsleben" },
        pt: { title: "Como posso ajudar?", subtitle: "Pergunte-me sobre a vida profissional de Bolaji" },
    };

    const renderSuggestions = (lang) => {
        if (!suggestionsGrid) return;
        while (suggestionsGrid.firstChild) suggestionsGrid.removeChild(suggestionsGrid.firstChild);

        const items = suggestions[lang] || suggestions.en;
        items.forEach(text => {
            const btn = document.createElement('button');
            btn.classList.add('suggestion-card');
            btn.textContent = text;
            btn.addEventListener('click', () => {
                userInput.value = text;
                chatForm.dispatchEvent(new Event('submit', { cancelable: true }));
            });
            suggestionsGrid.appendChild(btn);
        });

        // Update welcome text
        const texts = welcomeTexts[lang] || welcomeTexts.en;
        const titleEl = document.getElementById('welcome-title');
        const subtitleEl = document.getElementById('welcome-subtitle');
        if (titleEl) titleEl.textContent = texts.title;
        if (subtitleEl) subtitleEl.textContent = texts.subtitle;
    };

    // --- Placeholder per language ---
    const placeholders = {
        en: "Ask about Bolaji\u2019s professional life\u2026",
        fr: "Posez une question sur la vie pro de Bolaji\u2026",
        es: "\u00bfPreguntas sobre la vida profesional de Bolaji?",
        de: "Fragen \u00fcber Bolajis Berufsleben\u2026",
        pt: "Pergunte sobre a vida profissional de Bolaji\u2026",
        it: "Chiedi della vita professionale di Bolaji\u2026",
        ru: "\u0421\u043f\u0440\u043e\u0441\u0438\u0442\u0435 \u043e \u043f\u0440\u043e\u0444\u0435\u0441\u0441\u0438\u043e\u043d\u0430\u043b\u044c\u043d\u043e\u0439 \u0436\u0438\u0437\u043d\u0438 Bolaji\u2026",
        zh: "\u95ee\u5173\u4e8eBolaji\u7684\u804c\u4e1a\u751f\u6daf\u2026",
        ja: "Bolaji\u306e\u8077\u696d\u751f\u6d3b\u306b\u3064\u3044\u3066\u8cea\u554f\u2026",
        ko: "Bolaji\uc758 \uc9c1\uc5c5 \uc0dd\ud65c\uc5d0 \ub300\ud574 \ubb3c\uc5b4\ubcf4\uc138\uc694\u2026",
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

                if (isEmbedMode) {
                    // Embed: type welcome as a bubble, no suggestions
                    if (msgs.length > 0) {
                        await new Promise(r => setTimeout(r, 400));
                        await typeMessage(msgs[0], 'bot', { noFeedback: true });
                    }
                } else {
                    // Standalone: show suggestion cards
                    renderSuggestions(userLanguage);
                }
            } else {
                throw new Error('Welcome failed');
            }
        } catch (e) {
            userInput.placeholder = placeholders.en;
            if (isEmbedMode) {
                await new Promise(r => setTimeout(r, 400));
                await typeMessage(
                    "Hello! I\u2019m iBola, Bolaji\u2019s AI assistant. Ask me about his professional life.",
                    'bot', { noFeedback: true }
                );
            } else {
                renderSuggestions('en');
            }
        }

        userInput.focus();
    };

    init();
});

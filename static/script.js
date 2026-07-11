// ============================================================
// iBola Chatbot, portfolio design system frontend
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
    const railSuggestions = document.getElementById('rail-suggestions');
    const answerSourcesEl = document.getElementById('answer-sources');
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

    // --- Analytics helper ---
    const trackEvent = (name, params) => {
        if (typeof gtag === 'function') gtag('event', name, params);
    };

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
            const newTheme = document.body.classList.contains('dark-mode') ? 'light' : 'dark';
            applyTheme(newTheme);
            trackEvent('theme_toggle', { theme: newTheme });
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
    const createFeedbackRow = (idx, traceId) => {
        // Trace-level thumbs only make sense when the turn has a trace
        // (deterministic intents and cache hits carry no trace_id): rendering
        // them anyway would POST feedback that the backend 400s.
        if (!traceId) return document.createDocumentFragment();
        const fb = document.createElement('div');
        fb.classList.add('msg-feedback');
        fb.dataset.traceId = traceId;

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

    // One-tap reason chips shown inline only after a thumbs-down (no modal,
    // no required text). Values must match the /feedback allow-list.
    const REASON_CHIPS = [
        ['wrong-info', 'Wrong info'],
        ['didnt-answer', "Didn't answer"],
        ['too-vague', 'Too vague'],
        ['not-relevant', 'Not relevant'],
    ];

    const postScore = async (scoreName, value, traceId, comment) => {
        try {
            await fetch('/feedback', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    score_name: scoreName,
                    value: value,
                    session_id: sessionId,
                    trace_id: traceId || null,
                    comment: comment || null,
                }),
            });
        } catch (e) { /* silent: feedback is best-effort */ }
    };

    const showReasonChips = (container, traceId) => {
        const wrap = document.createElement('div');
        wrap.classList.add('feedback-reasons');
        REASON_CHIPS.forEach(([val, label]) => {
            const chip = document.createElement('button');
            chip.classList.add('reason-chip');
            chip.textContent = label;
            chip.addEventListener('click', () => {
                postScore('user-thumbs-reason', val, traceId);
                while (container.firstChild) container.removeChild(container.firstChild);
                const thanks = document.createElement('span');
                thanks.classList.add('feedback-thanks');
                thanks.textContent = 'Thanks for your feedback';
                container.appendChild(thanks);
            });
            wrap.appendChild(chip);
        });
        container.appendChild(wrap);
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
            trackEvent('chat_started', { session_id: sessionId, is_embed: isEmbedMode });
        }

        const { row, content } = createMessageRow(sender);

        if (sender === 'bot' && !opts.plain) {
            content.appendChild(renderMarkdown(text));
        } else {
            content.textContent = text;
        }

        if (sender === 'bot' && !opts.noFeedback) {
            content.appendChild(createFeedbackRow(msgIndex++, opts.traceId));
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
                        content.appendChild(createFeedbackRow(msgIndex++, opts.traceId));
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
        const traceId = container.dataset.traceId || null;

        container.querySelectorAll('.feedback-btn').forEach(b => {
            b.classList.remove('active');
            b.disabled = true;
        });
        btn.classList.add('active');
        trackEvent('feedback_given', {
            session_id: sessionId,
            type: score > 0 ? 'thumbs_up' : 'thumbs_down',
        });

        // Explicit BOOLEAN thumbs score attached to this turn's trace
        postScore('user-thumbs', score, traceId);

        if (score > 0) {
            setTimeout(() => {
                while (container.firstChild) container.removeChild(container.firstChild);
                const thanks = document.createElement('span');
                thanks.classList.add('feedback-thanks');
                thanks.textContent = 'Thanks for your feedback';
                container.appendChild(thanks);
            }, 600);
        } else {
            // Thumbs-down: reveal reason chips inline for a one-tap reason
            showReasonChips(container, traceId);
        }
    };

    // --- Action Buttons ---
    const addActionButtons = (actions) => {
        const group = document.createElement('div');
        group.classList.add('action-group');

        const icons = {
            contact_email: '\u2709\uFE0F',
            contact_booking: '\uD83D\uDCC5',
        };

        actions.forEach(action => {
            const btn = document.createElement('button');
            btn.classList.add('action-btn');
            if (action.primary) btn.classList.add('action-btn--primary');
            const icon = icons[action.type] || '';
            btn.textContent = icon ? `${icon}  ${action.text}` : action.text;
            btn.title = action.description || '';

            btn.addEventListener('click', () => {
                trackEvent('action_button_clicked', {
                    action_type: action.type,
                    action_text: action.text,
                    session_id: sessionId,
                });
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
    // Session-end CSAT (😞😐🙂), shown once for meaningful sessions only
    // (>=2 user exchanges), never on every visit. Attaches at session level.
    let csatShown = false;
    const maybeShowCsat = () => {
        if (csatShown || msgIndex < 2) return;
        csatShown = true;
        const wrap = document.createElement('div');
        wrap.classList.add('csat-row');
        const label = document.createElement('span');
        label.classList.add('csat-label');
        label.textContent = 'How was this chat?';
        wrap.appendChild(label);
        [['1', '😞'], ['2', '😐'], ['3', '🙂']].forEach(
            ([val, emoji]) => {
                const b = document.createElement('button');
                b.classList.add('csat-btn');
                b.textContent = emoji;
                b.setAttribute('aria-label', `Rating ${val} of 3`);
                b.addEventListener('click', () => {
                    postScore('session-csat', val, null);
                    wrap.querySelectorAll('.csat-btn').forEach(x => (x.disabled = true));
                    b.classList.add('active');
                    label.textContent = 'Thanks!';
                });
                wrap.appendChild(b);
            }
        );
        chatMessages.insertBefore(wrap, typingIndicator);
        scrollToBottom();
    };

    const endChat = () => {
        chatEnded = true;
        chatApp.classList.add('chat-ended');
        userInput.disabled = true;
        userInput.placeholder = 'Chat ended';
        sendBtn.disabled = true;
        trackEvent('chat_ended', { session_id: sessionId });
        maybeShowCsat();
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
        trackEvent('message_sent', {
            message_length: text.length,
            session_id: sessionId,
            is_embed: isEmbedMode,
        });

        try {
            const res = await fetch('/ask-agentic', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    user_input: text,
                    session_id: sessionId,
                    user_language: userLanguage,
                    stream: false,
                }),
            });

            hideTyping();

            if (res.ok) {
                const data = await res.json();
                // Retain the rated turn's trace_id so feedback attaches to it
                if (data.answer) await typeMessage(data.answer, 'bot', { traceId: data.trace_id });
                if (data.actions && data.actions.length > 0) addActionButtons(data.actions);
                renderAnswerSources(data.answer_sources);
                if (data.should_end_chat && !chatEnded) endChat();
                trackEvent('bot_response', {
                    intent: data.intent || 'unknown',
                    has_actions: !!(data.actions && data.actions.length > 0),
                    session_id: sessionId,
                });
            } else {
                addMessage('Something went wrong. Please try again.', 'bot', { noFeedback: true });
                trackEvent('bot_error', { status: res.status });
            }
        } catch (err) {
            hideTyping();
            addMessage('Connection error. Please check your internet and try again.', 'bot', { noFeedback: true });
            trackEvent('bot_error', { error: 'network' });
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
    const suggestions = [
        "What are Bolaji's key skills?",
        "Tell me about his work experience",
        "What is his educational background?",
        "How can I contact Bolaji?",
    ];

    const welcomeTexts = {
        title: "How can I help you?",
        subtitle: "Ask me anything about Bolaji's professional life",
    };

    const askSuggestion = (text) => {
        trackEvent('suggestion_clicked', { suggestion_text: text });
        userInput.value = text;
        chatForm.dispatchEvent(new Event('submit', { cancelable: true }));
    };

    const renderSuggestions = () => {
        const items = suggestions;

        // Welcome-card grid (shown on mobile / narrow viewports).
        if (suggestionsGrid) {
            while (suggestionsGrid.firstChild) suggestionsGrid.removeChild(suggestionsGrid.firstChild);
            items.forEach(text => {
                const btn = document.createElement('button');
                btn.classList.add('suggestion-card');
                btn.textContent = text;
                btn.addEventListener('click', () => askSuggestion(text));
                suggestionsGrid.appendChild(btn);
            });
        }

        // Left-rail editorial list (shown on desktop >=1100px).
        if (railSuggestions) {
            while (railSuggestions.firstChild) railSuggestions.removeChild(railSuggestions.firstChild);
            items.forEach(text => {
                const li = document.createElement('li');
                const btn = document.createElement('button');
                btn.classList.add('rail-suggestion');
                btn.textContent = text;
                btn.addEventListener('click', () => askSuggestion(text));
                li.appendChild(btn);
                railSuggestions.appendChild(li);
            });
        }

        // Update welcome text
        const texts = welcomeTexts;
        const titleEl = document.getElementById('welcome-title');
        const subtitleEl = document.getElementById('welcome-subtitle');
        if (titleEl) titleEl.textContent = texts.title;
        if (subtitleEl) subtitleEl.textContent = texts.subtitle;
    };

    // --- Answer sources rail (PART 8.2) ---
    // Render the normalized answer_sources receipts into the right rail so
    // recruiters see which profile sections grounded the answer, linked to
    // bolablg.com. No-op in embed (the rail element does not exist there).
    const renderAnswerSources = (sources) => {
        if (!answerSourcesEl) return;
        while (answerSourcesEl.firstChild) answerSourcesEl.removeChild(answerSourcesEl.firstChild);

        if (!Array.isArray(sources) || sources.length === 0) {
            const hint = document.createElement('p');
            hint.className = 'rail-hint';
            hint.textContent = 'This answer did not draw on a specific profile section.';
            answerSourcesEl.appendChild(hint);
            return;
        }

        sources.forEach(src => {
            const a = document.createElement('a');
            a.className = 'rail-source';
            a.href = src.url || 'https://www.bolablg.com';
            a.target = '_blank';
            a.rel = 'noopener';
            a.addEventListener('click', () => {
                trackEvent('answer_source_clicked', { source_id: src.id, session_id: sessionId });
            });

            const label = document.createElement('span');
            label.className = 'rail-source-label';
            const name = document.createElement('span');
            name.textContent = src.label || 'Profile';
            label.appendChild(name);
            // retrieval_rank is already 1-based; show it directly, and skip the
            // sentinel used for evidence that carried no rank.
            if (Number.isInteger(src.rank) && src.rank > 0 && src.rank < 100) {
                const rank = document.createElement('span');
                rank.className = 'rail-source-rank';
                rank.textContent = `#${src.rank}`;
                label.appendChild(rank);
            }
            a.appendChild(label);

            if (src.section) {
                const section = document.createElement('span');
                section.className = 'rail-source-section';
                section.textContent = src.section;
                a.appendChild(section);
            }
            answerSourcesEl.appendChild(a);
        });
    };

    // --- Placeholder per language ---
    // --- Init ---
    const init = async () => {
        userLanguage = 'en';
        userInput.placeholder = "Ask about Bolaji\u2019s professional life\u2026";

        if (isEmbedMode) {
            await new Promise(r => setTimeout(r, 400));
            await typeMessage(
                "Hello! I\u2019m iBola, Bolaji\u2019s AI assistant. Ask me about his professional life.",
                'bot', { noFeedback: true }
            );
        } else {
            renderSuggestions();
        }

        userInput.focus();
    };

    init();
});

// === MAIN IBOLA CHATBOT APPLICATION ===

document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 iBola Chatbot Application Starting...');

    // DOM Elements
    const chatBox = document.getElementById('chat-box');
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const typingIndicator = document.getElementById('typing-indicator');
    const sendButton = chatForm.querySelector('button');
    const modeToggle = document.getElementById('mode-toggle');

    // Multi-agent elements
    const agentIndicator = document.getElementById('agent-indicator');
    const agentIcon = document.getElementById('agent-icon');
    const agentName = document.getElementById('agent-name');
    const agentStatus = document.getElementById('agent-status');
    const quickActions = document.getElementById('quick-actions');

    // Modal elements
    const modal = document.getElementById('appointment-modal');
    const modalIframe = document.getElementById('modal-iframe');
    const closeModalBtn = document.querySelector('.modal-close-btn');
    const agentTransitionModal = document.getElementById('agent-transition-modal');

    // Session and state management
    const sessionId = `web-session-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    let lastUserQuestion = "";
    let currentAgentType = 'redirect';
    let agentTransitionTimeout = null;
    let userLanguage = 'en';
    let redirectCount = 0;

    // Agent configurations
    const agentConfigs = {
        professional: {
            icon: '💼',
            name: 'Professional Expert',
            status: 'Specialized in career & projects',
            color: '#007aff'
        },
        education: {
            icon: '🎓',
            name: 'Education Specialist',
            status: 'Focused on academic background',
            color: '#34c759'
        },
        learning: {
            icon: '📚',
            name: 'Learning Advisor',
            status: 'Guides skill development',
            color: '#ff9500'
        },
        redirect: {
            icon: '🔄',
            name: 'Smart Redirect',
            status: 'Routes to appropriate agent',
            color: '#ff3b30'
        }
    };

    console.log('📋 Elements loaded:', {
        chatBox: !!chatBox,
        chatForm: !!chatForm,
        userInput: !!userInput,
        modeToggle: !!modeToggle,
        sessionId: sessionId
    });

    // === THEME TOGGLE FUNCTIONALITY ===
    const applyTheme = (theme) => {
        console.log('🎨 applyTheme called with:', theme);
        const isDark = theme === 'dark';
        document.body.classList.toggle('dark-mode', isDark);

        // Update toggle visual elements
        const toggleIcon = modeToggle?.querySelector('.toggle-icon');
        const toggleText = modeToggle?.querySelector('.toggle-text');
        const toggleHandle = modeToggle?.querySelector('.toggle-handle');
        const toggleSlider = modeToggle?.querySelector('.toggle-slider');

        if (isDark) {
            if (toggleIcon) toggleIcon.textContent = '☀️';
            if (toggleText) toggleText.textContent = 'Light';
            if (toggleHandle) toggleHandle.style.left = '22px';
            if (toggleSlider) toggleSlider.style.background = 'linear-gradient(135deg, #63b3ed, #4299e1)';
            console.log('🌙 Switched to DARK mode');
        } else {
            if (toggleIcon) toggleIcon.textContent = '🌙';
            if (toggleText) toggleText.textContent = 'Dark';
            if (toggleHandle) toggleHandle.style.left = '2px';
            if (toggleSlider) toggleSlider.style.background = 'linear-gradient(135deg, #e9ecef, #dee2e6)';
            console.log('☀️ Switched to LIGHT mode');
        }

        localStorage.setItem('theme', theme);
    };

    // Theme toggle event listener
    if (modeToggle) {
        modeToggle.addEventListener('click', () => {
            console.log('🖱️ Toggle clicked!');
            const newTheme = document.body.classList.contains('dark-mode') ? 'light' : 'dark';
            console.log('🎨 Switching to theme:', newTheme);
            applyTheme(newTheme);

            // Add click animation
            modeToggle.style.animation = 'none';
            setTimeout(() => {
                modeToggle.style.animation = '';
            }, 100);
        });
        console.log('✅ Theme toggle event listener attached');
    }

    // === MESSAGE HANDLING ===
    const addMessage = (text, sender, question = null) => {
        console.log('📝 addMessage called:', { text, sender });

        // Create message wrapper for proper positioning
        const wrapper = document.createElement('div');
        wrapper.classList.add('message-wrapper', `${sender}-message-wrapper`);

        // Create message element
        const messageElement = document.createElement('div');
        messageElement.classList.add('message', `${sender}-message`);
        messageElement.textContent = text;

        // Add to DOM
        wrapper.appendChild(messageElement);
        chatBox.insertBefore(wrapper, typingIndicator);
        chatBox.scrollTop = chatBox.scrollHeight;

        console.log('✅ Message added to chat box');
    };

    const typeMessage = (text, sender = 'bot', question = null, speed = 40) => {
        console.log('📝 typeMessage called with:', { text, sender, speed });
        return new Promise(resolve => {
            typingIndicator.style.display = 'flex';

            // Create message wrapper for proper positioning
            const wrapper = document.createElement('div');
            wrapper.classList.add('message-wrapper', `${sender}-message-wrapper`);

            // Create message element
            const messageElement = document.createElement('div');
            messageElement.classList.add('message', `${sender}-message`);
            wrapper.appendChild(messageElement);

            chatBox.insertBefore(wrapper, typingIndicator);
            chatBox.scrollTop = chatBox.scrollHeight;

            let index = 0;
            const interval = setInterval(() => {
                messageElement.textContent += text.charAt(index);
                index++;
                chatBox.scrollTop = chatBox.scrollHeight;
                if (index >= text.length) {
                    clearInterval(interval);
                    typingIndicator.style.display = 'none';
                    resolve();
                }
            }, speed);
        });
    };

    // === AGENT MANAGEMENT ===
    const switchAgent = (newAgentType, showTransition = true) => {
        if (currentAgentType === newAgentType) return;

        const oldAgentType = currentAgentType;
        currentAgentType = newAgentType;

        // Update UI classes for theming
        document.body.classList.remove(`agent-${oldAgentType}`);
        document.body.classList.add(`agent-${newAgentType}`);

        // Update agent indicator (if exists)
        if (agentIcon && agentName && agentStatus) {
            const config = agentConfigs[newAgentType];
            if (config) {
                agentIcon.textContent = config.icon;
                agentName.textContent = config.name;
                agentStatus.textContent = config.status;
                agentIndicator.style.borderColor = config.color;
            }
        }

        // Show transition modal if requested
        if (showTransition) {
            showAgentTransition(newAgentType);
        }

        // Update quick actions visibility
        if (quickActions) {
            quickActions.style.display = newAgentType === 'redirect' ? 'flex' : 'none';
        }

        console.log(`🔄 Switched from ${oldAgentType} to ${newAgentType} agent`);
    };

    const showAgentTransition = (agentType) => {
        const transitionIcon = document.getElementById('transition-icon');
        const transitionTitle = document.getElementById('transition-title');
        const transitionMessage = document.getElementById('transition-message');

        if (transitionIcon && transitionTitle && transitionMessage) {
            const config = agentConfigs[agentType];
            if (config) {
                transitionIcon.textContent = config.icon;
                transitionTitle.textContent = config.name;
                transitionMessage.textContent = `Switching to ${config.name.toLowerCase()}...`;
            }

            if (agentTransitionModal) {
                agentTransitionModal.style.display = 'flex';

                if (agentTransitionTimeout) {
                    clearTimeout(agentTransitionTimeout);
                }

                agentTransitionTimeout = setTimeout(() => {
                    agentTransitionModal.style.display = 'none';
                }, 1500);
            }
        }
    };

    // === FORM SUBMISSION ===
    if (chatForm) {
        chatForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            console.log('📝 Form submitted!');

            const messageText = userInput.value.trim();
            if (!messageText) return;

            // Handle special commands
            if (["no", "non", "nein"].includes(messageText.toLowerCase())) {
                userInput.value = '';
                // endChat(); // Would need to implement this
                return;
            }

            lastUserQuestion = messageText;

            // Add user message
            addMessage(messageText, 'user');
            userInput.value = '';
            typingIndicator.style.display = 'flex';
            chatBox.scrollTop = chatBox.scrollHeight;

            console.log('📤 Sending message to backend...');

            try {
                // Send to backend
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        message: messageText,
                        session_id: sessionId,
                        language: userLanguage,
                        agent_type: currentAgentType
                    })
                });

                if (response.ok) {
                    const data = await response.json();
                    console.log('📥 Received response:', data);

                    // Switch agent if needed
                    if (data.agent_type && data.agent_type !== currentAgentType) {
                        switchAgent(data.agent_type, true);
                    }

                    // Display bot response
                    if (data.response) {
                        await typeMessage(data.response, 'bot');
                    }

                    // Handle actions if present
                    if (data.actions && data.actions.length > 0) {
                        // addActionButtons(data.actions); // Would need to implement this
                    }
                } else {
                    console.error('❌ Backend error:', response.status);
                    await typeMessage('Sorry, I encountered an error. Please try again.', 'bot');
                }
            } catch (error) {
                console.error('❌ Network error:', error);
                await typeMessage('Sorry, I\'m having trouble connecting. Please check your internet connection and try again.', 'bot');
            }

            typingIndicator.style.display = 'none';
        });
        console.log('✅ Form submission handler attached');
    }

    // === INITIALIZATION ===
    const initializeChatbot = async () => {
        console.log('🚀 Starting chatbot initialization...');

        try {
            // Get browser language and fetch localized welcome messages
            const browserLanguage = navigator.language || 'en';
            console.log('🌐 Browser language detected:', browserLanguage);

            const response = await fetch('/welcome', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: sessionId,
                    browser_language: browserLanguage
                })
            });

            console.log('📡 Welcome API response status:', response.status);

            if (response.ok) {
                const data = await response.json();
                userLanguage = data.detected_language;
                const messages = data.welcome_messages || [];

                console.log('✅ Welcome data received:', data);

                if (messages.length > 0) {
                    console.log('💬 Welcome message:', messages[0]);
                    console.log('📝 Displaying welcome message...');
                    await new Promise(res => setTimeout(res, 500));
                    await typeMessage(messages[0], 'bot');
                }

                // Update placeholder text based on detected language
                updatePlaceholderText(userLanguage);
            } else {
                console.log('⚠️ Welcome API failed, using fallback');
                const fallbackMessage = "Hello! I'm iBola, your AI assistant for Bolaji's professional journey.";
                await new Promise(res => setTimeout(res, 500));
                await typeMessage(fallbackMessage, 'bot');
                updatePlaceholderText('en');
            }

        } catch (error) {
            console.error('❌ Initialization error:', error);
            const fallbackMessage = "Hello! I'm iBola, your AI assistant for Bolaji's professional journey.";
            await new Promise(res => setTimeout(res, 500));
            await typeMessage(fallbackMessage, 'bot');
            updatePlaceholderText('en');
        }
    };

    // Function to update placeholder text based on language
    const updatePlaceholderText = (language) => {
        console.log('🔤 updatePlaceholderText called with language:', language);

        const placeholders = {
            'en': "What's up?",
            'fr': "Quoi de neuf ?",
            'es': "¿Qué pasa?",
            'de': "Was ist los?",
            'it': "Che succede?",
            'pt': "E aí?",
            'ru': "Что нового?",
            'zh': "有什么新鲜事吗？",
            'ja': "どうしたの？",
            'ko': "무슨 일이야?"
        };

        const placeholder = placeholders[language] || placeholders['en'];
        console.log('📝 Setting placeholder to:', placeholder);
        if (userInput) {
            userInput.placeholder = placeholder;
        }
    };

    // === MODAL HANDLERS ===
    if (closeModalBtn) {
        closeModalBtn.onclick = () => {
            if (modal) modal.style.display = "none";
            if (modalIframe) modalIframe.src = "";
        };
    }

    if (window) {
        window.onclick = (event) => {
            if (event.target == modal || event.target == agentTransitionModal) {
                if (modal) modal.style.display = "none";
                if (modalIframe) modalIframe.src = "";
            }
        };
    }

    // === QUICK ACTIONS ===
    document.querySelectorAll('.quick-action-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const agentType = btn.dataset.agent;
            console.log('🎯 Quick action clicked:', agentType);
            switchAgent(agentType, true);

            const simulatedQuery = getAgentQuery(agentType);
            if (simulatedQuery && userInput) {
                userInput.value = simulatedQuery;
                if (chatForm) {
                    chatForm.dispatchEvent(new Event('submit'));
                }
            }
        });
    });

    const getAgentQuery = (agentType) => {
        const queries = {
            professional: "Tell me about your professional experience",
            education: "What is your educational background?",
            learning: "How can I learn the skills you have?"
        };
        return queries[agentType] || "";
    };

    // === STARTUP ===
    console.log('🎯 All event listeners attached, starting initialization...');

    // Initialize theme
    const savedTheme = localStorage.getItem('theme') || 'light';
    applyTheme(savedTheme);

    // Start chatbot
    initializeChatbot();

    console.log('✅ iBola Chatbot Application Ready!');
});

document.addEventListener('DOMContentLoaded', () => {
    const chatBox = document.getElementById('chat-box');
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const typingIndicator = document.getElementById('typing-indicator');
    const sendButton = chatForm.querySelector('button');
    const modeToggle = document.getElementById('mode-toggle');

    const modal = document.getElementById('appointment-modal');
    const modalIframe = document.getElementById('modal-iframe');
    const closeModalBtn = document.querySelector('.modal-close-btn');

    const sessionId = `web-session-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    let lastUserQuestion = "";
    
    let rejectionInactivityTimerId = null;
    let globalInactivityTimerId = null;

    const REJECTION_MESSAGE = "I am trained to answer questions about Bolaji's professional background. Please ask a relevant question.";

    const endChat = () => {
        userInput.disabled = true;
        userInput.placeholder = "Chat ended.";
        sendButton.classList.add('disabled');
        if (rejectionInactivityTimerId) clearTimeout(rejectionInactivityTimerId);
        if (globalInactivityTimerId) clearTimeout(globalInactivityTimerId);
    };

    const resetGlobalInactiveTimer = () => {
        if (globalInactivityTimerId) clearTimeout(globalInactivityTimerId);
        globalInactivityTimerId = setTimeout(endChat, 60000);
    };

    const sendFeedback = async (question, answer, rating) => {
        try {
            await fetch('/feedback', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: sessionId, question, answer, rating })
            });
        } catch (error) {
            console.error("Failed to send feedback:", error);
        }
    };

    const addMessage = (text, sender, question = null) => {
        const wrapper = document.createElement('div');
        wrapper.classList.add('message-wrapper', `${sender}-message-wrapper`);
        const messageElement = document.createElement('div');
        messageElement.classList.add('message', `${sender}-message`);
        messageElement.textContent = text;
        wrapper.appendChild(messageElement);
        if (sender === 'bot' && question) {
            const feedbackContainer = document.createElement('div');
            feedbackContainer.classList.add('feedback-container');
            const trigger = document.createElement('span');
            trigger.classList.add('feedback-trigger');
            trigger.textContent = '🤔';
            const choices = document.createElement('div');
            choices.classList.add('feedback-choices', 'hidden');
            const likeBtn = document.createElement('span');
            likeBtn.textContent = '👍';
            likeBtn.onclick = () => {
                sendFeedback(question, text, 1);
                feedbackContainer.innerHTML = '<span class="feedback-thanks">Thanks for your feedback!</span>';
            };
            const dislikeBtn = document.createElement('span');
            dislikeBtn.textContent = '👎';
            dislikeBtn.onclick = () => {
                sendFeedback(question, text, -1);
                feedbackContainer.innerHTML = '<span class="feedback-thanks">Thanks for your feedback!</span>';
            };
            choices.appendChild(likeBtn);
            choices.appendChild(dislikeBtn);
            trigger.onclick = () => {
                choices.classList.toggle('hidden');
                trigger.style.display = 'none';
            };
            feedbackContainer.appendChild(trigger);
            feedbackContainer.appendChild(choices);
            wrapper.appendChild(feedbackContainer);
        }
        chatBox.insertBefore(wrapper, typingIndicator);
        chatBox.scrollTop = chatBox.scrollHeight;
    };

    const addActionButtons = (actions) => {
        const buttonContainer = document.createElement('div');
        buttonContainer.classList.add('message', 'bot-message');
        actions.forEach(action => {
            let button;
            if (action.type === 'popup') {
                button = document.createElement('button');
                button.textContent = action.text;
                button.onclick = () => {
                    modalIframe.src = action.url;
                    modal.style.display = 'flex';
                };
            } else {
                button = document.createElement('a');
                button.href = action.url;
                if (action.url.startsWith('http')) {
                    button.target = '_blank';
                }
                button.textContent = action.text;
            }
            button.classList.add('action-button');
            buttonContainer.appendChild(button);
        });
        chatBox.insertBefore(buttonContainer, typingIndicator);
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    closeModalBtn.onclick = () => { modal.style.display = "none"; modalIframe.src = ""; }
    window.onclick = (event) => {
        if (event.target == modal) {
            modal.style.display = "none";
            modalIframe.src = "";
        }
    }

    // --- THEME TOGGLE LOGIC ---
    const applyTheme = (theme) => {
        document.body.classList.toggle('dark-mode', theme === 'dark');
        modeToggle.textContent = theme === 'dark' ? '☀️' : '🌙';
        localStorage.setItem('theme', theme);
    };

    modeToggle.addEventListener('click', () => {
        const newTheme = document.body.classList.contains('dark-mode') ? 'light' : 'dark';
        applyTheme(newTheme);
    });

    // --- FORM SUBMISSION LOGIC ---
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        resetGlobalInactiveTimer();
        if (rejectionInactivityTimerId) {
            clearTimeout(rejectionInactivityTimerId);
            rejectionInactivityTimerId = null;
        }
        const messageText = userInput.value.trim();
        if (!messageText) return;
        if (["no", "non", "nein"].includes(messageText.toLowerCase())) {
            userInput.value = '';
            endChat();
            return;
        }
        lastUserQuestion = messageText;
        addMessage(messageText, 'user');
        userInput.value = '';
        typingIndicator.style.display = 'flex';
        chatBox.scrollTop = chatBox.scrollHeight;
        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ user_input: messageText, session_id: sessionId })
            });
            if (!response.ok) throw new Error('Network response was not ok');
            const data = await response.json();
            if (data.answer) {
                addMessage(data.answer, 'bot', lastUserQuestion);
                if (data.answer === REJECTION_MESSAGE) {
                    setTimeout(() => {
                        addMessage("Do you have another question?", 'bot');
                        rejectionInactivityTimerId = setTimeout(endChat, 15000);
                    }, 30000);
                }
            }
            if (data.actions) {
                addActionButtons(data.actions);
            }
        } catch (error) {
            console.error('Error:', error);
            addMessage('Sorry, something went wrong. Please try again later.', 'bot');
        } finally {
            typingIndicator.style.display = 'none';
        }
    });

    // --- INITIALIZATION ---
    const welcomeMessages = {
        en: ["Hello! I'm iBola.", "I can answer questions about Bolaji's professional background. How can I help you today?"],
        fr: ["Bonjour ! Je suis iBola.", "Je peux répondre aux questions sur le parcours professionnel de Bolaji. Comment puis-je vous aider aujourd'hui ?"],
        es: ["¡Hola! Soy iBola.", "Puedo responder preguntas sobre la trayectoria profesional de Bolaji. ¿Cómo puedo ayudarte hoy?"]
    };
    const userLang = navigator.language.split('-')[0];
    const messages = welcomeMessages[userLang] || welcomeMessages.en;
    setTimeout(() => addMessage(messages[0], 'bot'), 500);
    setTimeout(() => addMessage(messages[1], 'bot'), 1200);

    const savedTheme = localStorage.getItem('theme') || 'light';
    applyTheme(savedTheme);

    resetGlobalInactiveTimer();
});

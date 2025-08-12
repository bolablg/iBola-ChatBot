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

    const REJECTION_MESSAGE = "I'm here to help with questions about Bolaji's professional background. Feel free to ask anything relevant! 🙂";

    const endChat = () => {
        userInput.disabled = true;
        userInput.placeholder = "Chat ended.";
        sendButton.classList.add('disabled');
    };

    

    let afterResponseInactiveTimer = null;

    const endChatDueToInactivity = () => {
        addMessage("Ciao...✌🏿", 'bot');
        endChat();
    };

    const resetAfterResponseInactiveTimer = () => {
        if (afterResponseInactiveTimer) clearTimeout(afterResponseInactiveTimer);
        afterResponseInactiveTimer = setTimeout(endChatDueToInactivity, 180000);
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
            trigger.textContent = '💬';

            const choices = document.createElement('div');
            choices.classList.add('feedback-choices', 'hidden');

            const up = document.createElement('span');
            up.textContent = '👍';
            const down = document.createElement('span');
            down.textContent = '👎';

            const sendFeedback = async (rating) => {
                try {
                    await fetch('/feedback', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            session_id: sessionId,
                            question: question,
                            answer: text,
                            rating: rating
                        })
                    });
                    feedbackContainer.innerHTML = '<span class="feedback-thanks">Thanks for the feedback!</span>';
                } catch (err) {
                    console.error('Feedback error:', err);
                    feedbackContainer.innerHTML = '<span class="feedback-thanks">Feedback failed</span>';
                }
            };

            trigger.addEventListener('click', () => {
                choices.classList.toggle('hidden');
            });
            up.addEventListener('click', () => sendFeedback('up'));
            down.addEventListener('click', () => sendFeedback('down'));

            choices.appendChild(up);
            choices.appendChild(down);
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

    userInput.addEventListener('input', () => {
        if (afterResponseInactiveTimer) clearTimeout(afterResponseInactiveTimer);
    });

    // --- FORM SUBMISSION LOGIC ---
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        if (afterResponseInactiveTimer) clearTimeout(afterResponseInactiveTimer);
        
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
                resetAfterResponseInactiveTimer();
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
});

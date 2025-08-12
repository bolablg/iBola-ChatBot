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

    const typeMessage = (text, sender = 'bot', question = null, speed = 40) => {
        return new Promise(resolve => {
            typingIndicator.style.display = 'flex';
            const wrapper = document.createElement('div');
            wrapper.classList.add('message-wrapper', `${sender}-message-wrapper`);
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

    const endChatDueToInactivity = () => {
        typeMessage("Ciao...✌🏿", 'bot').then(endChat);
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
                await typeMessage(data.answer, 'bot', lastUserQuestion);
                resetAfterResponseInactiveTimer();
            } else {
                typingIndicator.style.display = 'none';
            }
            if (data.actions) {
                addActionButtons(data.actions);
            }
        } catch (error) {
            console.error('Error:', error);
            await typeMessage('Sorry, something went wrong. Please try again later.', 'bot');
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
    const showWelcomeMessages = async () => {
        await new Promise(res => setTimeout(res, 500));
        await typeMessage(messages[0], 'bot');
        await new Promise(res => setTimeout(res, 700));
        await typeMessage(messages[1], 'bot');
    };
    showWelcomeMessages();

    const savedTheme = localStorage.getItem('theme') || 'light';
    applyTheme(savedTheme);
});

// === MINIMAL DEBUG VERSION ===
console.log('🔧 Script loaded successfully!');

document.addEventListener('DOMContentLoaded', () => {
    console.log('🔧 DOM Content Loaded!');

    // Test 1: Check if elements exist
    const chatBox = document.getElementById('chat-box');
    const userInput = document.getElementById('user-input');
    const modeToggle = document.getElementById('mode-toggle');
    const chatForm = document.getElementById('chat-form');
    const toggleIcon = modeToggle?.querySelector('.toggle-icon');
    const toggleText = modeToggle?.querySelector('.toggle-text');

    console.log('📋 Element existence check:', {
        chatBox: !!chatBox,
        userInput: !!userInput,
        modeToggle: !!modeToggle,
        chatForm: !!chatForm,
        toggleIcon: !!toggleIcon,
        toggleText: !!toggleText
    });

    // Test 2: Toggle functionality with actual theme switching
    let isDarkMode = false;

    if (modeToggle) {
        modeToggle.addEventListener('click', () => {
            console.log('🖱️ TOGGLE CLICKED!');
            isDarkMode = !isDarkMode;

            // Toggle dark mode class on body
            document.body.classList.toggle('dark-mode', isDarkMode);

            // Update toggle visual elements
            if (toggleIcon && toggleText) {
                if (isDarkMode) {
                    toggleIcon.textContent = '☀️';
                    toggleText.textContent = 'Light';
                    console.log('🌙 Switched to DARK mode');
                } else {
                    toggleIcon.textContent = '🌙';
                    toggleText.textContent = 'Dark';
                    console.log('☀️ Switched to LIGHT mode');
                }
            }
        });
        console.log('✅ Toggle event listener attached successfully');
    } else {
        console.log('❌ Toggle element not found');
    }

    // Test 3: Message display
    if (chatBox) {
        console.log('💬 Adding test message...');

        // Create message wrapper (for positioning)
        const wrapper = document.createElement('div');
        wrapper.className = 'message-wrapper bot-message-wrapper';

        // Create bot message element
        const testMsg = document.createElement('div');
        testMsg.className = 'message bot-message';
        testMsg.textContent = '✅ JavaScript is working! This is a test message.';

        // Add message to wrapper, then wrapper to chat box
        wrapper.appendChild(testMsg);
        chatBox.appendChild(wrapper);
        console.log('✅ Test message added to chat box');
    } else {
        console.log('❌ Chat box element not found');
    }

    // Test 4: Placeholder
    if (userInput) {
        userInput.placeholder = '✅ Placeholder set! What\'s up?';
        console.log('✅ Placeholder updated successfully');
    } else {
        console.log('❌ User input element not found');
    }

    // Test 5: Form submission handling
    if (chatForm && userInput && chatBox) {
        chatForm.addEventListener('submit', (e) => {
            e.preventDefault();
            console.log('📝 Form submitted!');

            const messageText = userInput.value.trim();
            if (messageText) {
                console.log('💬 User message:', messageText);

                // Create message wrapper (for positioning)
                const wrapper = document.createElement('div');
                wrapper.className = 'message-wrapper user-message-wrapper';

                // Create user message element
                const userMessage = document.createElement('div');
                userMessage.className = 'message user-message';
                userMessage.textContent = messageText;

                // Add message to wrapper, then wrapper to chat box
                wrapper.appendChild(userMessage);
                chatBox.appendChild(wrapper);
                chatBox.scrollTop = chatBox.scrollHeight;

                // Clear input
                userInput.value = '';
                console.log('✅ User message displayed and input cleared');
            } else {
                console.log('⚠️ Empty message, not sending');
            }
        });
        console.log('✅ Form submission handler attached');
    } else {
        console.log('❌ Form or required elements not found for message handling');
    }

    console.log('🎯 All basic tests completed! Check browser for results.');
});

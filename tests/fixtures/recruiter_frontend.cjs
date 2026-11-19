const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('static/script.js', 'utf8');

function extract(start, end) {
    return source.slice(source.indexOf(start), source.indexOf(end, source.indexOf(start)));
}
class Element {
    constructor(tag) { this.tag = tag; this.children = []; this.classList = {add() {}}; this.style = {}; }
    appendChild(child) { this.children.push(child); }
    setAttribute(key, value) { this[key] = value; }
    addEventListener() {}
}
(async () => {
    const handlers = {};
    const requests = [];
    const messages = [];
    let resolveRequest;
    const state = {
        chatForm: {addEventListener(name, fn) { handlers[name] = fn; }},
        userInput: {value: 'First evidence question', style: {}},
        sendBtn: {disabled: false},
        isEmbedMode: false, sessionId: 'regression', userLanguage: 'en', chatEnded: false, requestInFlight: false,
        addMessage(text, sender, opts) { messages.push({text, sender, opts}); },
        typeMessage(text, sender, opts) { messages.push({text, sender, opts}); },
        showTyping() {}, hideTyping() {}, trackEvent() {}, addActionButtons() {}, endChat() {},
        fetch(...args) { requests.push(args); return new Promise(resolve => {resolveRequest = resolve;}); },
        document: {createElement(tag) {return new Element(tag);}}, URL, Promise,
    };
    vm.createContext(state);
    vm.runInContext(extract("    chatForm.addEventListener('submit'", '    // --- Modal ---'), state);
    const event = {preventDefault() {}};
    const first = handlers.submit(event);
    state.userInput.value = 'Second question';
    await handlers.submit(event);
    assert.equal(requests.length, 1, 'A second submit must not send overlapping history');
    assert.equal(state.sendBtn.disabled, true);
    resolveRequest({ok: true, json: async () => ({answer: 'Verified answer', answer_sources: [{label:'Confirmed scope'}]})});
    await first;
    assert.equal(state.sendBtn.disabled, false);
    assert.equal(messages.filter(m => m.sender === 'bot').length, 1);
    assert.equal(messages.at(-1).opts.answerSources[0].label, 'Confirmed scope');
    const retry = handlers.submit(event);
    resolveRequest({ok: false, status: 503});
    await retry;
    assert.equal(state.requestInFlight, false, 'Errors must release the submit guard');

    vm.runInContext(extract('    // Keep receipts beside their answer', '    // --- Rotating profile quotes') + '\nthis.renderSources = renderAnswerSources;', state);
    const receipts = new Element('div');
    state.renderSources([
        {label:'Public project', url:'https://www.bolablg.com/projects/ibola/'},
        {label:'Owner-confirmed scope', url:null},
        {label:'Unsafe link', url:'javascript:alert(1)'},
    ], receipts);
    assert.equal(receipts.children[0].tag, 'a');
    assert.equal(receipts.children[1].tag, 'span', 'Private records must not get an invented homepage URL');
    assert.equal(receipts.children[2].tag, 'span');

    vm.runInContext(extract('    // Display the completed', '    // --- Feedback ---') + '\nthis.display = typeMessage;', state);
    state.hideTyping = () => {};
    await state.display('A long verified answer '.repeat(100));
    assert.equal(messages.at(-1).text.length, 2300, 'The completed answer must appear without per-character timers');
    console.log('Frontend recruiter regressions passed');
})().catch(error => { console.error(error); process.exitCode = 1; });

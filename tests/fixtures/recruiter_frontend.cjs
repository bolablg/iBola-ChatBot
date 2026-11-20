const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('static/script.js', 'utf8');

function extract(start, end) {
    return source.slice(source.indexOf(start), source.indexOf(end, source.indexOf(start)));
}
class Element {
    constructor(tag) {
        this.tag = tag;
        this.children = [];
        this.classes = new Set();
        this.classList = {add: (...names) => names.forEach(name => this.classes.add(name))};
        this.style = {};
        this.dataset = {};
    }
    appendChild(child) { this.children.push(child); }
    insertBefore(child) { this.children.push(child); }
    setAttribute(key, value) { this[key] = value; }
    addEventListener() {}
}
function descendants(element) {
    return [element, ...element.children.flatMap(descendants)];
}
function renderedText(element) {
    return (element.textContent || '') + element.children.map(renderedText).join('');
}
(async () => {
    const handlers = {};
    const requests = [];
    const messages = [];
    let resolveRequest;
    const state = {
        chatForm: {addEventListener(name, fn) { handlers[name] = fn; }},
        userInput: {value: "What are Bolaji's key skills?", style: {}},
        sendBtn: {disabled: false},
        isEmbedMode: false, sessionId: 'regression', userLanguage: 'en', chatEnded: false, requestInFlight: false,
        addMessage(text, sender, opts) { messages.push({text, sender, opts}); },
        typeMessage(text, sender, opts) { messages.push({text, sender, opts}); },
        showTyping() {}, hideTyping() {}, trackEvent() {}, addActionButtons() {}, endChat() {},
        fetch(...args) { requests.push(args); return new Promise(resolve => {resolveRequest = resolve;}); },
        document: {
            createElement(tag) {return new Element(tag);},
            createElementNS(_ns, tag) {return new Element(tag);},
            createTextNode(text) {const node = new Element('#text'); node.textContent = text; return node;},
            createDocumentFragment() {return new Element('#fragment');},
        }, URL, Promise,
    };
    const screenshotAnswer = "Bolaji's key skills include Python as his primary programming language, advanced SQL, BigQuery, Google Cloud Platform, Airflow, Dataform, AI and LLM engineering, FinOps, and data team leadership.";
    const sources = [
        {label: 'Skills & tools', section: 'CORE EXPERTISE AREAS:', url: 'https://www.bolablg.com/skills/'},
        {label: 'Skills & tools', section: 'LEADERSHIP & SOFT SKILLS:', url: 'https://www.bolablg.com/skills/'},
        {label: 'Education', section: 'CONTINUOUS PROFESSIONAL DEVELOPMENT:', url: 'https://www.bolablg.com/llms-full.txt'},
        {label: 'Skills & tools', section: 'ADDITIONAL SKILLS:', url: 'https://www.bolablg.com/skills/'},
        {label: 'Owner-confirmed services', section: 'PRIVATE INTERNAL HEADING', url: null},
        {label: 'Unsafe link', section: 'INTERNAL RESOURCE', url: 'javascript:alert(1)'},
    ];
    vm.createContext(state);
    vm.runInContext(extract("    chatForm.addEventListener('submit'", '    // --- Modal ---'), state);
    const event = {preventDefault() {}};
    const first = handlers.submit(event);
    state.userInput.value = 'Second question';
    await handlers.submit(event);
    assert.equal(requests.length, 1, 'A second submit must not send overlapping history');
    assert.equal(state.sendBtn.disabled, true);
    resolveRequest({ok: true, json: async () => ({answer: screenshotAnswer, answer_sources: sources, trace_id: 'rated-turn'})});
    await first;
    assert.equal(state.sendBtn.disabled, false);
    assert.equal(messages.filter(m => m.sender === 'bot').length, 1);
    assert.equal(messages.at(-1).text, screenshotAnswer);
    assert.equal(messages.at(-1).opts.traceId, 'rated-turn');
    assert.equal(messages.at(-1).opts.answerSources, undefined, 'Source metadata must not enter the display options');
    const retry = handlers.submit(event);
    resolveRequest({ok: false, status: 503});
    await retry;
    assert.equal(state.requestInFlight, false, 'Errors must release the submit guard');

    vm.runInContext(extract('    // Display the completed', '    // --- Feedback ---') + '\nthis.display = typeMessage;', state);
    state.hideTyping = () => {};
    await state.display('A long verified answer '.repeat(100));
    assert.equal(messages.at(-1).text.length, 2300, 'The completed answer must appear without per-character timers');

    state.chatMessages = new Element('main');
    state.typingIndicator = new Element('div');
    state.msgIndex = 0;
    state.hasUserSent = true;
    state.hideWelcomeScreen = () => {};
    state.scrollToBottom = () => {};
    vm.runInContext(extract('    // --- SVG icon builders', '    // One-tap reason chips'), state);
    vm.runInContext(extract('    // --- Create message row with avatar', '    // Display the completed') + '\nthis.renderMessage = addMessage; this.displayActual = typeMessage;', state);
    for (const embedded of [false, true]) {
        state.isEmbedMode = embedded;
        const content = await state.displayActual(screenshotAnswer, 'bot', {answerSources: sources, traceId: 'rated-turn'});
        assert.equal(renderedText(content), screenshotAnswer, 'Only the answer text should appear, in full chat and embed');
        const nodes = descendants(content);
        assert.equal(nodes.filter(node => node.tag === 'button').length, 2, 'Feedback must remain available');
        assert.equal(nodes.filter(node => node.tag === 'a').length, 0, 'Public source receipts must also stay hidden');
        assert.equal(nodes.filter(node => /source/.test(node.className || '')).length, 0);
        assert.equal(nodes.filter(node => node['aria-label'] === 'Answer sources').length, 0);
        for (const receipt of sources) {
            assert.equal(renderedText(content).includes(receipt.section), false, 'KB headings must not be rendered');
        }
    }
    const linked = state.renderMessage('Read the [iBola project](https://www.bolablg.com/projects/ibola/).', 'bot', {answerSources: sources});
    const links = descendants(linked).filter(node => node.tag === 'a');
    assert.equal(links.length, 1, 'An explicit answer link must survive hiding automatic metadata');
    assert.equal(links[0].href, 'https://www.bolablg.com/projects/ibola/');
    assert.equal(links[0].textContent, 'iBola project');
    console.log('Frontend recruiter regressions passed');
})().catch(error => { console.error(error); process.exitCode = 1; });

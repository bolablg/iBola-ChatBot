const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('static/script.js', 'utf8');
const start = source.indexOf('    // --- Optional first-hand context');
const end = source.indexOf('    // --- Contact Alert ---', start);
class Element {
    constructor(tag) { this.tag = tag; this.children = []; this.events = {}; this.classList = { add() {} }; }
    setAttribute(key, value) { this[key] = value; }
    appendChild(child) { child.parent = this; this.children.push(child); return child; }
    addEventListener(name, fn) { this.events[name] = fn; }
    remove() { this.parent.children = this.parent.children.filter(child => child !== this); }
    click() { if (this.events.click) this.events.click(); }
}
function setup(storage = new Map()) {
    const context = {
        document: { createElement: tag => new Element(tag) },
        sessionStorage: { getItem: key => storage.get(key), setItem: (key, value) => storage.set(key, value) },
        createSvg: () => new Element('svg'), URL, Set,
        fetch() { throw Error('No contact network request is allowed'); },
        sendContactAlert() { throw Error('No contact notification is allowed'); },
        scrollToBottom() { throw Error('No unsolicited scrolling is allowed'); },
        window: { open() { throw Error('Links must use native navigation'); } },
    };
    vm.createContext(context);
    vm.runInContext(source.slice(start, end) + '\nthis.render = renderContextualHandoff; this.actions = addActionButtons;', context);
    return context;
}
const action = {
    type: 'contextual_handoff', version: 1, heading: 'Need more context?', description: 'Ask Bolaji about the details not covered here.', dismiss_text: 'Not now',
    links: [
        {kind:'email', text:'Email Bolaji', description:'Open a draft', url:'mailto:hello@bolablg.com?subject=Follow-up&body=A%26B'},
        {kind:'booking', text:'Choose a time', description:'Open calendar', url:'https://calendar.app.google/KYm4B5iL5j3HRxav9'},
    ],
};
const storage = new Map();
const context = setup(storage);
const answer = new Element('div');
context.actions([action], answer);
assert.equal(answer.children.length, 1);
const card = answer.children[0];
const links = card.children[2];
assert.equal(links.children[0].tag, 'a');
assert.equal(links.children[0].href, action.links[0].url);
assert.equal(links.children[1].target, '_blank');
assert.equal(links.children[1].rel, 'noopener noreferrer');
links.children[0].click();
links.children[1].click();
assert.equal(links.children[2].tag, 'button');
assert.equal(links.children[2].type, 'button');
const second = new Element('div');
context.actions([action], second);
assert.equal(second.children.length, 0, 'Only one automatic offer per visit');
setup(storage).actions([action], second);
assert.equal(second.children.length, 0, 'Refresh does not repeat an offer');
links.children[2].click();
assert.equal(answer.children.length, 0);
assert.equal(storage.get('ibola.contextual-handoff.v1'), 'dismissed');
setup(storage).actions([action], second);
assert.equal(second.children.length, 0, 'Dismissal survives refresh');
for (const bad of ['javascript:alert(1)', 'https://calendar.app.google.evil.test/', 'https://user@calendar.app.google/']) {
    const mutated = JSON.parse(JSON.stringify(action)); mutated.links[1].url = bad;
    const clean = new Element('div'); setup().actions([mutated], clean); assert.equal(clean.children.length, 0);
}
for (const method of ['getItem', 'setItem']) {
    const broken = setup(); broken.sessionStorage[method] = () => { throw Error('Storage blocked'); };
    const clean = new Element('div'); broken.actions([action], clean); assert.equal(clean.children.length, 0);
}
const french = {...action, heading: 'Besoin de précisions ?', dismiss_text: 'Pas maintenant'};
const frenchAnswer = new Element('div'); setup().actions([french], frenchAnswer);
assert.equal(frenchAnswer.children[0].children[0].textContent, french.heading);
console.log('Contextual handoff frontend passed');

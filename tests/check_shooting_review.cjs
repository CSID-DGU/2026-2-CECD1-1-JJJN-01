const fs = require('fs');
const path = require('path');
const vm = require('vm');
const assert = require('assert');
const file = process.argv[2];
const html = fs.readFileSync(file, 'utf8');
assert(!/__CALIBRATION__|__COMMAND__|__ROWS__|__BASELINE__|__TRACKING__/.test(html));
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const context = new Proxy({}, {get: () => () => {}});
const elements = {plot: {getContext: () => context}, time: {value: 0}, label: {textContent: ''}};
vm.runInNewContext(script, {document: {getElementById: id => elements[id]}});
assert(Number(elements.time.max) > 100);
assert(elements.label.textContent.includes('\uacf5 \uc7a1\uae30'));
elements.time.value = elements.time.max;
elements.time.oninput();
assert(elements.label.textContent.includes('\ub9b4\ub9ac\uc2a4 \uc774\ud6c4'));
for (const match of html.matchAll(/(?:src|href)="([^"]+)"/g)) {
  assert(fs.existsSync(path.resolve(path.dirname(file), match[1])), `Missing asset: ${match[1]}`);
}
console.log('Review script, scrubber, placeholders, and all local assets passed.');

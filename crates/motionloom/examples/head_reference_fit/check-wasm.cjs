// =========================================
// =========================================
// crates/motionloom/examples/head_reference_fit/check-wasm.cjs

// Compare actual WASM execution with the independently generated native CLI report.
const fs = require('node:fs');
const assert = require('node:assert/strict');
const path = require('node:path');
const [modulePath, sourcePath, requestPath, nativeReportPath] = process.argv.slice(2);
if (!nativeReportPath) throw new Error('usage: node check-wasm.cjs MODULE.js SOURCE REQUEST NATIVE_REPORT');
const wasm = require(path.resolve(modulePath));
const source = fs.readFileSync(sourcePath, 'utf8');
const request = fs.readFileSync(requestPath, 'utf8');
const native = JSON.parse(fs.readFileSync(nativeReportPath, 'utf8'));
assert.equal(JSON.parse(wasm.validateHeadReferenceSet(request)).valid, true);
const evaluated = JSON.parse(wasm.evaluateHeadReferenceFit(source, request));
const fitted = JSON.parse(wasm.fitHeadAssetToReferences(source, request));
const tolerance = 2e-6;
for (const key of ['objective', 'widthOverHeight', 'depthOverHeight', 'meshHeight']) {
  assert.ok(Math.abs(evaluated[key] - native.beforeMetrics[key]) < tolerance, `before ${key}`);
  assert.ok(Math.abs(fitted.afterMetrics[key] - native.afterMetrics[key]) < tolerance, `after ${key}`);
}
assert.equal(wasm.applyHeadFitProposal(source, JSON.stringify(fitted)), fitted.candidateDsl);
assert.throws(() => wasm.applyHeadFitProposal(source + '\n', JSON.stringify(fitted)));
console.log(JSON.stringify({status: 'passed', tolerance, nativeBefore: native.beforeMetrics.objective,
  wasmBefore: evaluated.objective, nativeAfter: native.afterMetrics.objective, wasmAfter: fitted.afterMetrics.objective}));

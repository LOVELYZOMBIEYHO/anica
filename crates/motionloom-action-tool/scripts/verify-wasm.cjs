// =========================================
// =========================================
// crates/motionloom-action-tool/scripts/verify-wasm.cjs

// Execute the same diagnostic evaluator in WASM; no browser or GPU is required.
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const hash = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
const [pkg, model, bundlePath, output] = process.argv.slice(2);
if (!output) throw new Error('Usage: node verify-wasm.cjs <nodejs/motionloom.js> <target.glb> <bundle.json> <report.json>');
const inputPaths = [pkg, model, bundlePath].map(p => fs.realpathSync(p));
if (fs.existsSync(output) || inputPaths.includes(path.resolve(output))) throw new Error('Report must be a new, separate file');
const bundleBytes = fs.readFileSync(bundlePath);
const bundle = JSON.parse(bundleBytes);
const modelBytes = fs.readFileSync(model);
if (hash(modelBytes) !== bundle.target_sha256 || hash(bundle.world_dsl) !== bundle.world_sha256) throw new Error('Input hash mismatch');
if (!Array.isArray(bundle.snapshots) || !bundle.snapshots.length) throw new Error('No validation snapshots');
const { WasmPoseDiagnostics } = require(inputPaths[0]);
const evaluator = new WasmPoseDiagnostics(bundle.world_dsl, modelBytes);
let maximum = 0;
let compared = 0;
try {
  for (const expected of bundle.snapshots) {
    const actual = JSON.parse(evaluator.sample_json(expected.actor_id, expected.frame, expected.fps));
    if (actual.stage !== expected.stage || actual.joints.length !== expected.joints.length) throw new Error('Snapshot stage/count mismatch');
    for (let i = 0; i < actual.joints.length; i++) {
      const a = actual.joints[i], e = expected.joints[i];
      if (a.node_index !== e.node_index || a.node_name !== e.node_name || a.canonical_bone !== e.canonical_bone) throw new Error('Joint identity mismatch');
      for (let j = 0; j < 16; j++) {
        const difference = Math.abs(a.model_global_matrix[j] - e.model_global_matrix[j]);
        if (!Number.isFinite(difference)) throw new Error('Non-finite matrix');
        maximum = Math.max(maximum, difference);
      }
      compared++;
    }
  }
} finally { evaluator.free(); }
const report = {
  action_sha256: bundle.action_sha256, target_sha256: bundle.target_sha256,
  world_sha256: bundle.world_sha256, bundle_sha256: hash(bundleBytes),
  wasm_sha256: hash(fs.readFileSync(path.join(path.dirname(inputPaths[0]), 'motionloom_bg.wasm'))),
  stage: bundle.stage, samples: bundle.snapshots.length, compared_joints: compared,
  max_matrix_difference: maximum, tolerance: 0.00001, passed: maximum <= 0.00001,
};
fs.writeFileSync(output, JSON.stringify(report, null, 2) + '\n', { flag: 'wx' });
console.log(JSON.stringify(report, null, 2));
if (!report.passed) process.exitCode = 1;

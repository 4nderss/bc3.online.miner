// Runs the built module against the node's BC3Hash vectors and measures its
// hash rate - the wasm counterpart of `cargo test` in bc3hash/, because the
// host tests check the host build of the C++, not the wasm32 one. A compiler
// can get mix.cpp wrong for one target only (GCC 13 did, see
// bc3hash/build.rs), so the module that ships is the one that is tested.
//
//   node wasm/test/vectors.mjs <module.wasm> <vectors dir> [bench seconds]
//
// The vectors dir is the node's doc/bc3/vectors (BC3HASH_VECTORS in the Rust
// tests). Exit status 0 only if every check passed.
import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { join } from "node:path";

const [wasmPath, vectorsDir, benchArg] = process.argv.slice(2);
if (!wasmPath || !vectorsDir) {
  console.error("usage: node vectors.mjs <module.wasm> <vectors dir> [bench seconds]");
  process.exit(2);
}
const benchSeconds = Number(benchArg ?? 5);

let failures = 0;
const check = (ok, what) => {
  if (!ok) failures++;
  console.log(`${ok ? "ok  " : "FAIL"} ${what}`);
};
const hex = (b) => Buffer.from(b).toString("hex");
const unhex = (s) => Uint8Array.from(Buffer.from(s, "hex"));

const bytes = readFileSync(wasmPath);
const module = new WebAssembly.Module(bytes);
// The worker instantiates it with no imports at all; anything imported here
// would fail in the browser.
const imports = WebAssembly.Module.imports(module);
check(imports.length === 0, `no imports (${imports.map((i) => `${i.module}.${i.name}`).join(", ") || "none"})`);
const { exports: w } = new WebAssembly.Instance(module, {});
console.log(`module: ${bytes.length} bytes`);
// The page refuses to start unless this is 1: the chain-verified genesis
// block hash, through the module's header serialization and SHA256d.
check(w.self_test() === 1, "self_test() = 1 (the genesis block)");

const t0 = performance.now();
const init = w.bc3hash_init();
const initMs = performance.now() - t0;
check(init === 1, `bc3hash_init() = ${init} in ${initMs.toFixed(0)} ms (tables + the node's 4 self-test vectors)`);
if (init !== 1) process.exit(1);
console.log(`memory after init: ${(w.memory.buffer.byteLength / 2 ** 20).toFixed(1)} MiB`);

const buf = (n) => w.alloc(n);
const view = () => new Uint8Array(w.memory.buffer);
const out = buf(32);
const header = buf(80);
const target = buf(32);

function bytesInto(which, data) {
  const p = data.length ? buf(data.length) : 0;
  if (data.length) view().set(data, p);
  const rc = w.bc3hash_bytes_into(which, p, data.length, out);
  if (data.length) w.dealloc(p, data.length);
  return rc === 1 ? hex(view().slice(out, out + 32)) : `rc=${rc}`;
}

// reference-vectors.tsv: input, bc3mix, bc3hash from the node's frozen Python
// reference - the empty input, the unfolded 64-byte path, folded long inputs.
const tsv = readFileSync(join(vectorsDir, "reference-vectors.tsv"), "utf8");
let n = 0;
for (const line of tsv.split(/\r?\n/)) {
  if (!line.trim() || line.startsWith("#")) continue;
  const [input, mix, hash] = line.split("\t");
  const data = input === "-" ? new Uint8Array() : unhex(input);
  const label = input === "-" ? "(empty)" : input.length > 16 ? `${input.slice(0, 16)}... (${data.length} B)` : input;
  check(bytesInto(1, data) === mix, `bc3mix  ${label}`);
  check(bytesInto(0, data) === hash, `bc3hash ${label}`);
  n++;
}
check(n >= 4, `${n} reference vectors`);

// header-vectors.json: one header under all four combinations of bits 11 and
// 12, through the page's path - version -> job_pow_algo -> hash_header_algo.
// With bit 11 it is BC3Hash and must give the node's digest; without it the
// module refuses the job (-1) and hashes nothing, as the CLI does: SHA256d
// (id 0) and SHA3-256t (id 1) are not mined since the second fork.
const headers = JSON.parse(readFileSync(join(vectorsDir, "header-vectors.json"), "utf8"));
for (const v of headers) {
  const h = unhex(v.header);
  const version = new DataView(h.buffer).getUint32(0, true);
  const name = `header version ${version.toString(16).padStart(8, "0")}`;
  const algo = w.job_pow_algo(version);
  view().set(h, header);
  if (version & 0x800) {
    const rc = w.hash_header_algo(algo, header, out);
    const got = hex(view().slice(out, out + 32));
    check(algo === 2 && rc === 1 && got === v.digest, `${name} -> algo ${algo}: ${got}`);
  } else {
    const refused = w.hash_header_algo(0, header, out) === 0 && w.hash_header_algo(1, header, out) === 0;
    check(algo === -1 && refused, `${name} -> algo ${algo}, not hashed`);
  }
}
check(headers.length === 4, `${headers.length} header vectors`);

// Cross-check against the native build: the same 64 headers hashed by the
// host build of bc3hash give CROSS_CHECK (wasm/src/lib.rs, test
// `cross_check_matches_the_wasm_harness`).
const CROSS_CHECK = "0b625bcc9887a866a75689b8b1a91f21a5c0fee4b3ce7c319aef84e146a3ebeb";
const agg = createHash("sha256");
for (let i = 0; i < 64; i++) {
  agg.update(Buffer.from(bytesInto(0, crossHeader(i)), "hex"));
}
check(agg.digest("hex") === CROSS_CHECK, "64 pseudo-random headers agree with the native build");

function crossHeader(i) {
  const h = new Uint8Array(80);
  for (let j = 0; j < 80; j++) h[j] = (j * 31 + i * 7 + 0x5a) & 0xff;
  new DataView(h.buffer).setUint32(0, 0x20001800, true);
  new DataView(h.buffer).setUint32(76, i, true);
  return h;
}

// search_algo finds exactly the first nonce a per-nonce loop finds, at a
// loose target (1/16 of the space), and nothing in a range that ends before.
{
  const h = crossHeader(1000);
  view().set(h, header);
  const t = new Uint8Array(32).fill(0xff);
  t[0] = 0x0f;
  view().set(t, target);
  let expected = -1;
  for (let nonce = 0; nonce < 400 && expected < 0; nonce++) {
    new DataView(w.memory.buffer).setUint32(header + 76, nonce, true);
    w.hash_header_algo(2, header, out);
    const d = view().slice(out, out + 32);
    // little-endian number vs big-endian target
    let cmp = 0;
    for (let k = 0; k < 32 && cmp === 0; k++) cmp = d[31 - k] - t[k];
    if (cmp <= 0) expected = nonce;
  }
  view().set(h, header);
  const got = w.search_algo(2, header, target, 0, 400);
  check(expected >= 0 && got === expected, `search_algo(BC3Hash) hit nonce ${got}, per-nonce loop ${expected}`);
  check(w.search_algo(2, header, target, 0, expected) === -1, "no hit before it");
}

// Hash rate, one thread, as the worker calls it: batches of search_algo with
// an impossible target.
if (benchSeconds > 0) {
  view().set(crossHeader(7), header);
  view().set(new Uint8Array(32), target);
  let hashes = 0;
  const start = performance.now();
  while (performance.now() - start < benchSeconds * 1000) {
    w.search_algo(2, header, target, hashes, 8);
    hashes += 8;
  }
  const s = (performance.now() - start) / 1000;
  console.log(`bench: ${hashes} BC3Hash in ${s.toFixed(1)} s = ${(hashes / s).toFixed(1)} H/s (${((s / hashes) * 1000).toFixed(2)} ms/hash), one thread`);
}

console.log(failures ? `${failures} FAILED` : "all passed");
process.exit(failures ? 1 : 0);

//! A process that ends while other threads are inside a hash must end
//! cleanly. The node's mix.cpp keeps its secp256k1 context and its tables in
//! function statics, which the C++ runtime destroys at exit; the miner never
//! joins its workers, and the 2.1.0 release workflow's test binary died of
//! SIGSEGV after every test had passed. See `HASHING` in src/lib.rs.

use std::process::Command;

/// Set in the child process this test starts: hash on eight threads, then
/// exit from under them.
const CHILD: &str = "BC3HASH_EXIT_WHILE_HASHING";

#[test]
fn the_process_exits_cleanly_while_threads_hash() {
    if std::env::var_os(CHILD).is_some() {
        bc3hash::initialize().expect("BC3Hash");
        for t in 0..8u8 {
            std::thread::spawn(move || {
                let mut header = [t; 80];
                for nonce in 0u32.. {
                    header[76..].copy_from_slice(&nonce.to_le_bytes());
                    std::hint::black_box(bc3hash::bc3hash(&header));
                }
            });
        }
        std::thread::sleep(std::time::Duration::from_millis(300));
        // Runs the exit handlers and the C++ destructors with all eight
        // threads hashing.
        std::process::exit(0);
    }
    for run in 0..5 {
        let status = Command::new(std::env::current_exe().expect("test binary"))
            .args(["--exact", "the_process_exits_cleanly_while_threads_hash", "--test-threads=1"])
            .env(CHILD, "1")
            .status()
            .expect("start the child");
        assert!(status.success(), "run {run}: the process died at exit ({status})");
    }
}

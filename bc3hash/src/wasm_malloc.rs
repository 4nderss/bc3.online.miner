//! malloc, calloc, realloc, aligned_alloc and free for the C++ on wasm32, on
//! Rust's allocator. The Rust target (wasm32-unknown-unknown) has no C
//! library, and WASI's would bring a second heap growing the same linear
//! memory; one heap is simpler to reason about and costs nothing here - the
//! reference code allocates its tables once and nothing per hash.
//!
//! Every block is preceded by a 16-byte header (16 is the alignment malloc
//! promises, max_align_t on wasm32) holding its size and alignment, so that
//! `free` can rebuild the `Layout`. A block aligned above 16 (mix.cpp's
//! curve tables are alignas(64)) starts `align` bytes into its allocation,
//! which leaves room for the header and keeps the alignment.

use std::alloc::{alloc, alloc_zeroed, dealloc, Layout};

const HEADER: usize = 16;

/// The allocation behind a block of `size` bytes aligned to `align`, and the
/// offset of the block in it.
fn layout(size: usize, align: usize) -> Option<(Layout, usize)> {
    let offset = align.max(HEADER);
    Some((Layout::from_size_align(size.checked_add(offset)?, offset).ok()?, offset))
}

unsafe fn allocate(size: usize, align: usize, zeroed: bool) -> *mut u8 {
    let Some((l, offset)) = layout(size, align) else {
        return core::ptr::null_mut();
    };
    let base = if zeroed { alloc_zeroed(l) } else { alloc(l) };
    if base.is_null() {
        return base;
    }
    let ptr = base.add(offset);
    let header = ptr.sub(HEADER) as *mut usize;
    header.write(size);
    header.add(1).write(align);
    ptr
}

/// Size and alignment of a block from `allocate`.
unsafe fn header(ptr: *mut u8) -> (usize, usize) {
    let header = ptr.sub(HEADER) as *const usize;
    (header.read(), header.add(1).read())
}

#[no_mangle]
pub unsafe extern "C" fn malloc(size: usize) -> *mut u8 {
    allocate(size, HEADER, false)
}

#[no_mangle]
pub unsafe extern "C" fn calloc(count: usize, size: usize) -> *mut u8 {
    match count.checked_mul(size) {
        Some(n) => allocate(n, HEADER, true),
        None => core::ptr::null_mut(),
    }
}

/// For operator new of over-aligned types.
#[no_mangle]
pub unsafe extern "C" fn aligned_alloc(align: usize, size: usize) -> *mut u8 {
    if !align.is_power_of_two() {
        return core::ptr::null_mut();
    }
    allocate(size, align, false)
}

#[no_mangle]
pub unsafe extern "C" fn free(ptr: *mut u8) {
    if ptr.is_null() {
        return;
    }
    let (size, align) = header(ptr);
    let (l, offset) = layout(size, align).expect("a block allocate made");
    dealloc(ptr.sub(offset), l);
}

#[no_mangle]
pub unsafe extern "C" fn realloc(ptr: *mut u8, size: usize) -> *mut u8 {
    if ptr.is_null() {
        return malloc(size);
    }
    let (old, align) = header(ptr);
    let new = allocate(size, align, false);
    if !new.is_null() {
        core::ptr::copy_nonoverlapping(ptr, new, old.min(size));
        free(ptr);
    }
    new
}

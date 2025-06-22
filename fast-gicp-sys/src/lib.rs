//! Low-level FFI bindings for the fast_gicp C++ library.
//!
//! This crate provides unsafe bindings to the fast_gicp C++ library through the cxx crate.
//! For a safe, high-level API, use the `fast-gicp` crate instead.

pub use minimal_lib::ffi;

mod minimal_lib;

#[cfg(test)]
mod test_minimal;

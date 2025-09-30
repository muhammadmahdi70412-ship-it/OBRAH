/// # OBRAH
/// OBRAH (OpenCL But Rusty And High-level) is an API built on top of OBWIO. It provides easy ways to use GPUs for
/// GPGPU computing, meaning anyone can do it.
/// 
/// ## Modules
/// OBRAH has 3 modules:
/// - kernel
/// - data
/// - runtime
/// These three libraries are necessary in every OBRAH program.
/// ### Kernel:
/// The Kernel module provides functions for building kernels and running them.
/// ### Data:
/// The Data module allows for sending data to and from the GPU, and making buffers for it.
/// ### Runtime:
/// The Runtime module provides all the main functions for OBRAH; setup, cleanup, and program making.
pub mod runtime;
pub mod data;
pub mod kernel;
# OBRAH

OBRAH (OpenCL But Rusty And High-level) is a Rust crate built on top of [OBWIO](https://github.com/muhammadmahdi70412-ship-it/obwio)(crates.io: [OBWIO](https://crates.io/crates/obwio), providing a **high-level, ergonomic interface for OpenCL**. It abstracts low-level C bindings, letting you perform GPU computations with minimal boilerplate.

## Overview

OBRAH provides an easy-to-use API for:

* Creating OpenCL contexts and command queues
* Loading kernels from `.cl` files
* Allocating and managing GPU buffers
* Running kernels and retrieving results

It is ideal for Rust developers who want GPU acceleration without dealing with raw OpenCL FFI.

## Features

* High-level Rust interface for OpenCL
* Built on OBWIO for cross-platform GPU support
* Simple buffer management (`buffer\_write`, `to\_gpu`, `from\_gpu`)
* Kernel management (`use\_kernel`, `make\_prog`, `setarg`, `run\_kernel`)
* Cleanup utilities to safely release OpenCL resources

## Installation

Add OBRAH to your `Cargo.toml`:

```toml
\[dependencies]
obrah = "1.1.3"
```

OBRAH automatically includes OBWIO as a dependency.

## Requirements

* Rust 1.70 or later
* OpenCL SDK installed on your system
* Compatible OpenCL runtime and drivers

## License

Licensed under the Apache License, Version 2.0

## Contributing

Contributions are welcome! Open issues, submit pull requests, or improve the documentation.

## Acknowledgments

OBRAH is built on OBWIO, which uses bindgen to generate FFI bindings. Thanks to the Rust and OpenCL communities for their documentation and support.


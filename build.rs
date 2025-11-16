fn main() {
    // Try to read the environment variable OPENCL_SDK.
    if let Ok(path) = std::env::var("OPENCL_SDK") {
        println!("cargo:rustc-link-search=native={}/lib", path);
        println!("cargo:rustc-link-lib=dylib=OpenCL");
    } else {
        println!("cargo:warning=Environment variable OPENCL_SDK not set. Please set it to your OpenCL SDK path.");
    }
}
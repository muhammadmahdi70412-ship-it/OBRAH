fn main() {
    if cfg!(target_os = "linux") {
        // linux is easy
        println!("cargo:rustc-link-lib=OpenCL");
        // stupid windows is not
    } else {
        // i swear if you dont have this set i am just... READ THE DOCS OMGG
        if let Ok(path) = std::env::var("OPENCL_SDK") {
            println!("cargo:rustc-link-search=native={}/lib", path);
            println!("cargo:rustc-link-lib=dylib=OpenCL");
        } else {
            println!(
                "cargo:error=Environment variable OPENCL_SDK not set. Set it to your OpenCL SDK path."
            );
        }
    }
}

use obwio::*;
use std::ffi::CString;
use std::fs;

/// This structure holds all the data; the platform, the device, the variables, etc.
///
/// # Examples
///
/// ```rust
///
/// fn main() {
///     use obrah::runtime::Env;
///     let mut env = Env::new(0, 0);
///     env.use_kernel("examples/vecadd_kernel.cl");
///     env.program();
/// }
/// ``````
pub struct Env {
    pub platform: cl_platform_id,
    pub device: cl_device_id,
    pub context: cl_context,
    pub queue: cl_command_queue,
    pub program: cl_program,
    pub kernel: cl_kernel,
    pub kerncode: Option<String>,
    pub err: cl_int,
}

impl Env {
    /// Make the kernel.
    pub fn make_kernel(&mut self, name: &str) {
        unsafe {
            let cname = CString::new(name).unwrap();
            self.kernel = clCreateKernel(self.program, cname.as_ptr(), &mut self.err);
            if self.err != 0 {
                panic!("OpenCL error: {}", self.err);
            }
        }
    }

    /// new() creates a new Env.
    pub fn new(plat: usize, dev: usize) -> Self {
        setup(plat, dev)
    }
    /// program() programs and sets up the environment.
    pub fn program(&mut self) -> &mut Self {
        make_prog(self);
        self
    }
    /// use_kernel() uses a kernel from a path.
    pub fn use_kernel(&mut self, path: &str) -> &mut Self {
        use_kernel(self, path);
        self
    }
}

impl Drop for Env {
    fn drop(&mut self) {
        cleanup(self);
    }
}

/// The setup() function sets the platform, device, context and queue and initialises everything.
/// Setup is only done on Env::new(), and you cannot call it by itself.
fn setup(plat: usize, dev: usize) -> Env {
    unsafe {
        let mut num_platforms: cl_uint = 0;
        clGetPlatformIDs(0, std::ptr::null_mut(), &mut num_platforms);

        let mut platforms: Vec<cl_platform_id> = vec![std::ptr::null_mut(); num_platforms as usize];
        clGetPlatformIDs(num_platforms, platforms.as_mut_ptr(), std::ptr::null_mut());

        let mut num_devices: cl_uint = 0;
        clGetDeviceIDs(
            platforms[plat],
            CL_DEVICE_TYPE_GPU.into(),
            0,
            std::ptr::null_mut(),
            &mut num_devices,
        );

        let mut devices: Vec<cl_device_id> = vec![std::ptr::null_mut(); num_devices as usize];
        clGetDeviceIDs(
            platforms[plat],
            CL_DEVICE_TYPE_GPU.into(),
            num_devices,
            devices.as_mut_ptr(),
            std::ptr::null_mut(),
        );

        let platform = platforms[plat];
        let device = devices[dev];

        let mut err: cl_int = 0;

        let context = clCreateContext(
            std::ptr::null_mut(),
            1,
            &device,
            None,
            std::ptr::null_mut(),
            &mut err,
        );

        let queue = clCreateCommandQueue(context, device, 0, &mut err);

        if err != 0 {
            panic!("OpenCL error: {}", err);
        }
        Env {
            platform,
            device,
            context,
            queue,
            program: std::ptr::null_mut(),
            kernel: std::ptr::null_mut(),
            kerncode: None,
            err,
        }
    }
}

// The make_prog() function uses the kernel and initialisations to make the actual OpenCL programs.
fn make_prog(env: &mut Env) {
    unsafe {
        let source = env.kerncode.as_ref().expect("Kernel not loaded!");

        let c_source = CString::new(source.as_str()).unwrap();
        let mut src_ptr = c_source.as_ptr();
        let program =
            clCreateProgramWithSource(env.context, 1, &mut src_ptr, std::ptr::null(), &mut env.err);

        clBuildProgram(
            program,
            1,
            &env.device,
            std::ptr::null_mut(),
            None,
            std::ptr::null_mut(),
        );
        if env.err != 0 {
            panic!("OpenCL error: {}", env.err);
        }
        env.program = program
    }
}

/// use_kernel() loads the kernel from a path.
fn use_kernel(env: &mut Env, path: &str) {
    let source = fs::read_to_string(path).expect("Failed to read kernel file");
    env.kerncode = Some(source);
}

/// cleanup() cleans the setup variables. This is used automatically by Drop for the Env struct.
/// It cannot be called on its own.
///
/// # Examples
///
/// ```rust
/// use obrah::runtime::Env;
///
/// fn main () {
///     let env = Env::new(0, 0);
/// } //<- automatically cleaned up
/// ```
fn cleanup(env: &mut Env) {
    unsafe {
        clReleaseKernel(env.kernel);
        clReleaseProgram(env.program);
        clReleaseCommandQueue(env.queue);
        clReleaseContext(env.context);
    }
}

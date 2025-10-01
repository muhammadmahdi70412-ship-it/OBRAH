use obwio::*;
use std::fs;
use std::ffi::CString;



/// This structure holds all the data; the platform, the device, the variables, etc.
/// 
/// # Examples
/// 
/// ```rust
/// 
/// fn main() {
///     use obrah::runtime::Env;
///     let mut env = Env::new();
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
    /// new() creates a new Env.
    pub fn new(gpu: bool) -> Self {
        setup(gpu)
    }
    /// program() programs and sets up the environment.
    pub fn program(&mut self) {
        make_prog(self);
    }
    /// use_kernel() uses a kernel from a path.
    pub fn use_kernel(&mut self, path: &str) {
        use_kernel(self, path);
    }
}

impl Drop for Env {
    fn drop(&mut self) {
        cleanup(self);
    }
}


/// The setup() function sets the platform, device, context and queue and initialises everything.
/// Setup is only done on Env::new(), and you cannot call it by itself.
fn setup(gpu: bool) -> Env {
    unsafe{
        if !gpu {
            let mut platform: cl_platform_id = std::ptr::null_mut();
            clGetPlatformIDs(1, &mut platform, std::ptr::null_mut());

            let mut device: cl_device_id = std::ptr::null_mut();
            clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU.into(), 1, &mut device, std::ptr::null_mut());

            let mut err: cl_int = 0;

    
            let context = clCreateContext(std::ptr::null_mut(), 1, &device, None, std::ptr::null_mut(), &mut err);

            let queue = clCreateCommandQueue(context, device, 0, &mut err);

            if err != 0 {
                panic!("OpenCL error: {}", err);
            }

            return Env {
                platform,
                device,
                context,
                queue,
                program: std::ptr::null_mut(),
                kernel: std::ptr::null_mut(),
                kerncode: None,
                err
            }
        } else {
            let mut num_platforms = 0;
            clGetPlatformIDs(0, std::ptr::null_mut(), &mut num_platforms);
            if num_platforms == 1 {
                panic!("You don't have a dGPU! If you are sure you do, then please wait for a future update while I fix it - it is a top priority.")
            }
            let mut platforms = vec![std::ptr::null_mut(); num_platforms as usize];
            clGetPlatformIDs(num_platforms, platforms.as_mut_ptr(), std::ptr::null_mut());
            let platform = platforms[1]; // second platform

            let mut device: cl_device_id = std::ptr::null_mut();
            clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU.into(), 1, &mut device, std::ptr::null_mut());

            let mut err: cl_int = 0;

    
            let context = clCreateContext(std::ptr::null_mut(), 1, &device, None, std::ptr::null_mut(), &mut err);

            let queue = clCreateCommandQueue(context, device, 0, &mut err);

            if err != 0 {
                panic!("OpenCL error: {}", err);
            }

            return Env {
                platform,
                device,
                context,
                queue,
                program: std::ptr::null_mut(),
                kernel: std::ptr::null_mut(),
                kerncode: None,
                err
            }

        }

    }

}




// The make_prog() function uses the kernel and initialisations to make the actual OpenCL programs.
fn make_prog(env: &mut Env) {
    unsafe {
        let source = env.kerncode.as_ref().expect("Kernel not loaded!");

        
        let c_source = CString::new(source.as_str()).unwrap();
        let mut src_ptr = c_source.as_ptr();
        let program = clCreateProgramWithSource(env.context, 1, &mut src_ptr, std::ptr::null(), &mut env.err);
        
        clBuildProgram(program, 1, &env.device, std::ptr::null_mut(), None, std::ptr::null_mut());
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
/// It cannot be called on it's own.
/// 
/// # Examples
/// 
/// ```rust
/// use obrah::runtime::Env;
/// 
/// fn main () {
///     let env = Env::new();
/// } //<- automatically cleaned up
/// ```
fn cleanup(env: &mut Env) {
    unsafe{
        clReleaseKernel(env.kernel);
        clReleaseProgram(env.program);
        clReleaseCommandQueue(env.queue);
        clReleaseContext(env.context);
    }
}



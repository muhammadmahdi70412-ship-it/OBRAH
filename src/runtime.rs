use obwio::*;
use std::fs;
use std::ffi::CString;


/// This structure holds all the data; the platform, the device, the variables, etc.
pub struct Env {
    pub platform: cl_platform_id,
    pub device: cl_device_id,
    pub context: cl_context,
    pub queue: cl_command_queue,
    pub program: cl_program,
    pub kernel: cl_kernel,
    pub buffers: Vec<cl_mem>,
    pub kerncode: Option<String>,
    pub err: cl_int,
}


/// The setup() function sets the platform, device, context and queue and initialises everything.
pub fn setup() -> Env {
    unsafe{
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

        Env {
            platform,
            device,
            context,
            queue,
            program: std::ptr::null_mut(),
            kernel: std::ptr::null_mut(),
            buffers: Vec::new(),
            kerncode: None,
            err
        }

    }
}



// The make_prog() function uses the kernel and initialisations to make the actual OpenCL Programs
pub fn make_prog(env: &mut Env) {
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

pub fn use_kernel(env: &mut Env, path: &str) {
    let source = fs::read_to_string(path).expect("Failed to read kernel file");
    env.kerncode = Some(source);
}

pub fn cleanup(env: &mut Env) {
    unsafe{
        clReleaseKernel(env.kernel);
        clReleaseProgram(env.program);
        clReleaseCommandQueue(env.queue);
        clReleaseContext(env.context);
    }
}

pub fn cleanvar (env: &mut Env, idx: usize) {
    unsafe {
        clReleaseMemObject(env.buffers[idx]);
    }
}

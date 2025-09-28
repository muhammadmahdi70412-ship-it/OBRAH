use obwio::*;
use std::fs;

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

pub fn setup() -> Env {
    unsafe{
        let mut platform: cl_platform_id = std::ptr::null_mut();
        clGetPlatformIDs(1, &mut platform, std::ptr::null_mut());

        let mut device: cl_device_id = std::ptr::null_mut();
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU.into(), 1, &mut device, std::ptr::null_mut());

        let mut err: cl_int = 0;
    
        let context = clCreateContext(std::ptr::null_mut(), 1, &device, None, std::ptr::null_mut(), &mut err);

        let queue = clCreateCommandQueue(context, device, 0, &mut err);

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

pub fn make_prog(env: &mut Env) {
    unsafe {
        let source = env.kerncode.as_ref().expect("Kernel not loaded!");

        let mut src_ptr = source.as_ptr() as *const i8;
        let src_len = source.len();

        let program = clCreateProgramWithSource(env.context, 1, &mut src_ptr, &(src_len as usize), &mut env.err);
        
        clBuildProgram(program, 1, &env.device, std::ptr::null_mut(), None, std::ptr::null_mut());

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

//obrah::main::use_kernel("vecadd.cl");
//obrah::main::setup();

//let bufa = obrah::data::buffer_write(&a);
//let bufb = obrah::data::buffer_write(&b);
//let bufc = obrah::data::buffer_write(&c);

//obrah::data::to_gpu(&bufa);
//obrah::data::to_gpu(&bufb);

//obrah::main::make_prog();
//obrah::kernel::make_kernel("vecAdd");

//obrah::kernel::setargs(&bufa, &bufb, &bufc, &n);
//obrah::kernel::run_kernel(N);

//obrah::data::from_gpu(&bufc, std::mem::size_of::<f32>() * N, &mut c);

//obrah::main::cleanup();
//obrah::main::cleanvar(bufa);
//obrah::main::cleanvar(bufb);
//obrah::main::cleanvar(bufc);

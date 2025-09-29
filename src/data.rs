use obwio::*;
use crate::runtime::Env;
use crate::runtime;


pub struct Buffer {
    pub buffer: cl_mem,
}

impl Buffer {
    pub fn to(&mut self, data: &[f32], env: &mut Env) {
        to_gpu(env, data, self);
    }
    pub fn from(&mut self, data: &mut [f32], env: &mut Env) {
        from_gpu(env, data,self);
    }
    pub fn new(env: & mut Env, data: &mut [f32]) -> Buffer {
        buffer_write(env, data)
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        runtime::cleanvar(self);
    }
}



/// Create a buffer in order to be sent to the GPU.
fn buffer_write(env: &mut Env, data: &[f32]) -> Buffer {
    unsafe {
        let size = data.len() * std::mem::size_of::<f32>();
        let buf = clCreateBuffer(env.context, CL_MEM_READ_WRITE.into(), size, std::ptr::null_mut(), &mut env.err);
        if env.err != 0 {
            panic!("OpenCL error: {}", env.err);
        }

        Buffer {
            buffer: buf,
        }
    }
}

/// Send the buffer, and the data supposed to be in the buffer, to the GPU.
fn to_gpu(env: &mut Env, data: &[f32], buffer: &mut Buffer) {
    unsafe {
        let size = data.len() * std::mem::size_of::<f32>();
        clEnqueueWriteBuffer(
            env.queue,
            buffer.buffer,
            CL_TRUE,
            0,
            size,
            data.as_ptr() as *const _,
            0,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            
        );
        if env.err != 0 {
            panic!("OpenCL error: {}", env.err);
        }
    }
}


/// Get data from the GPU, from a specific buffer.
fn from_gpu(env: &mut Env, data: &mut [f32], buf: &mut Buffer) {
    unsafe {
        let size = data.len() * std::mem::size_of::<f32>();
        clEnqueueReadBuffer(
            env.queue,
            buf.buffer,
            CL_TRUE,
            0,
            size,
            data.as_mut_ptr() as *mut _,
            0,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        );
        
        if env.err != 0 {
            panic!("OpenCL error: {}", env.err);
        }
    }
}

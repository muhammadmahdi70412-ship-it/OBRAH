use obwio::*;
use crate::runtime::Env;
use crate::runtime;


pub struct Buffer<'a> {
    pub buffer: usize,
    pub env: &'a mut Env,
}

impl<'a> Buffer<'a> {
    pub fn to(&mut self, data: &[f32]) {
        to_gpu(self.env, data, self.buffer);
    }
    pub fn from(&mut self, data: &mut [f32]) {
        from_gpu(self.env, data,self.buffer);
    }
    pub fn new(env: &'a mut Env, data: &mut [f32]) -> Buffer<'a> {
        buffer_write(env, data)
    }
}

impl<'a> Drop for Buffer<'a> {
    fn drop(&mut self) {
        runtime::cleanvar(self.env, self.buffer);
    }
}



/// Create a buffer in order to be sent to the GPU.
fn buffer_write<'a>(env: &'a mut Env, data: &[f32]) -> Buffer<'a> {
    unsafe {
        let size = data.len() * std::mem::size_of::<f32>();
        let buf = clCreateBuffer(env.context, CL_MEM_READ_WRITE.into(), size, std::ptr::null_mut(), &mut env.err);
        env.buffers.push(buf);
        if env.err != 0 {
            panic!("OpenCL error: {}", env.err);
        }
        let idx = env.buffers.len() - 1;

        Buffer {
            buffer: idx,
            env
        }
    }
}

/// Send the buffer, and the data supposed to be in the buffer, to the GPU.
fn to_gpu(env: &mut Env, data: &[f32], buffer: usize) {
    unsafe {
        let size = data.len() * std::mem::size_of::<f32>();
        clEnqueueWriteBuffer(
            env.queue,
            env.buffers[buffer],
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
fn from_gpu(env: &mut Env, data: &mut [f32], idx: usize) {
    unsafe {
        let size = data.len() * std::mem::size_of::<f32>();
        clEnqueueReadBuffer(
            env.queue,
            env.buffers[idx],
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

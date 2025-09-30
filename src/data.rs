use obwio::*;
use crate::runtime::Env;

/// Buffer is a struct that holds, well, a buffer.
/// It is returned by Buffer::new().
/// 
/// # Examples
/// 
/// ```rust
/// use crate::data::Buffer;
/// use crate::runtime::Env;
/// 
/// let mut env = Env::new();
/// let mut data = vec![1.0f32; 10];
/// let buf = Buffer::new(&mut env, &mut data);
/// ```
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
        cleanvar(self);
    }
}



/// Create a buffer in order to be sent to the GPU. This function is used implicitly by Buffer::new().
/// You don't have to call it.
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
/// Used by Buffer.to().
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
/// Used by Buffer.from().
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


/// cleanvar() cleans the buffer. This function is used automatically by Drop for the Buffer struct.
/// 
/// # Examples
/// 
/// ```rust
/// use crate::data::Buffer;
/// use crate::runtime::Env;
/// {
///     let mut env = Env::new();
///     let mut data = vec![1.0f32; 10];
///     let buf = Buffer::new(&mut env, &mut data);
/// } // <- automatically dropped here
/// ```
fn cleanvar (buf: &mut Buffer) {
    unsafe {
        clReleaseMemObject(buf.buffer);
    }
}
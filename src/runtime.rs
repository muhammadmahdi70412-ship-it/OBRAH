use obwio::*;
use std::ffi::CString;
use std::fmt;
use std::fs;

/// This structure holds all the data; the platform, the device, the variables, etc.
///
/// # Examples
///
/// ```rust
///
/// fn main() -> Result((), ClError) {
///     use obrah::runtime::{ClError, Env};
///     let mut env = Env::new(0, 0);
///     env.use_kernel("examples/vecadd_kernel.cl").program()?;
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

#[derive(Debug)]
pub enum ClError {
    InvalidProgram,
    InvalidKernelName,
    InvalidArgIndex,
    OutOfResources,
    OutOfMemory,
    InvalidContext,
    BuildProgramFailed,
    NonexistentPlatform,
    UnknownError(i32),
}

impl fmt::Display for ClError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidKernelName => {
                write!(f, "Kernel name not found")
            }
            Self::InvalidProgram => {
                write!(f, "Program not found")
            }
            Self::OutOfResources => {
                write!(f, "Out of resources")
            }
            Self::UnknownError(unkwn) => {
                write!(f, "Unknown error code: {unkwn}")
            }
            Self::InvalidContext => {
                write!(f, "Invalid context")
            }
            Self::OutOfMemory => {
                write!(f, "Out of memory")
            }
            Self::BuildProgramFailed => {
                write!(f, "Failed to build program.")
            }
            Self::NonexistentPlatform => {
                write!(f, "Platform out of range")
            }
            Self::InvalidArgIndex => {
                write!(f, "Invalid arg index")
            }
        }
    }
}

impl From<i32> for ClError {
    fn from(code: i32) -> Self {
        match code {
            -42 => ClError::InvalidProgram,
            -46 => ClError::InvalidKernelName,
            -5 => ClError::OutOfResources,
            -6 => ClError::OutOfMemory,
            -34 => ClError::InvalidContext,
            -11 => ClError::BuildProgramFailed,
            -49 => ClError::InvalidArgIndex,
            _ => ClError::UnknownError(code),
        }
    }
}

impl std::error::Error for ClError {}

impl Env {
    /// Make the kernel.
    pub fn make_kernel(&mut self, name: &str) -> Result<(), ClError> {
        unsafe {
            let cname = CString::new(name).unwrap();
            self.kernel = clCreateKernel(self.program, cname.as_ptr(), &mut self.err);
            if self.kernel.is_null() {
                Err(ClError::from(self.err))
            } else {
                Ok(())
            }
        }
    }

    /// new() creates a new Env.
    pub fn new(plat: usize, dev: usize) -> Result<Self, ClError> {
        setup(plat, dev)
    }
    /// program() programs and sets up the environment.
    pub fn program(&mut self) -> Result<&mut Self, ClError> {
        make_prog(self)?;
        Ok(self)
    }
    /// use_kernel() uses a kernel from a path.
    pub fn use_kernel(&mut self, path: &str) -> Result<&mut Self, Box<dyn std::error::Error>> {
        use_kernel(self, path)?;
        Ok(self)
    }
}

impl Drop for Env {
    fn drop(&mut self) {
        cleanup(self);
    }
}

/// The setup() function sets the platform, device, context and queue and initialises everything.
/// Setup is only done on Env::new(), and you cannot call it by itself.
fn setup(plat: usize, dev: usize) -> Result<Env, ClError> {
    unsafe {
        let mut num_platforms: cl_uint = 0;
        clGetPlatformIDs(0, std::ptr::null_mut(), &mut num_platforms);
        if plat >= num_platforms as usize {
            return Err(ClError::NonexistentPlatform);
        }

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

        if dev >= num_devices as usize {
            return Err(ClError::NonexistentPlatform);
        }

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

        if err != 0 {
            return Err(ClError::from(err));
        }

        let queue = clCreateCommandQueue(context, device, 0, &mut err);

        if err != 0 {
            return Err(ClError::from(err));
        }

        Ok(Env {
            platform,
            device,
            context,
            queue,
            program: std::ptr::null_mut(),
            kernel: std::ptr::null_mut(),
            kerncode: None,
            err,
        })
    }
}

// The make_prog() function uses the kernel and initialisations to make the actual OpenCL programs.
fn make_prog(env: &mut Env) -> Result<(), ClError> {
    unsafe {
        let source = env
            .kerncode
            .as_ref()
            .expect("Kernel not loaded! Call use_kernel first.");

        let c_source = CString::new(source.as_str()).unwrap();
        let mut src_ptr = c_source.as_ptr();
        let program =
            clCreateProgramWithSource(env.context, 1, &mut src_ptr, std::ptr::null(), &mut env.err);
        if program.is_null() {
            Err(ClError::from(env.err))
        } else {
            let builderr = clBuildProgram(
                program,
                1,
                &env.device,
                std::ptr::null_mut(),
                None,
                std::ptr::null_mut(),
            );
            if builderr != 0 {
                Err(ClError::from(builderr))
            } else {
                env.program = program;
                Ok(())
            }
        }
    }
}

/// use_kernel() loads the kernel from a path.
fn use_kernel(env: &mut Env, path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let source = fs::read_to_string(path)?;
    env.kerncode = Some(source);
    Ok(())
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

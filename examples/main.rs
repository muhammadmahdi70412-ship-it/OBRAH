use obrah::runtime;
use obrah::kernel;
use obrah::data;

fn main() {
    // Setup OpenCL environment
    let mut env = runtime::setup();

    // Load kernel source from file
    runtime::use_kernel(&mut env, "D:/ashha/testobrah/kernel.cl");
    
    // Build program
    runtime::make_prog(&mut env);
    
    // Create the kernel object from program
    kernel::make_kernel(&mut env, "vecAdd");

    // Sample data
    let a = vec![1.0f32, 2.0, 3.0, 4.0];
    let b = vec![10.0f32, 20.0, 30.0, 40.0];
    let mut c = vec![0.0f32; 4];

    // Allocate buffers on GPU
    let bufa = data::buffer_write(&mut env, &a);
    let bufb = data::buffer_write(&mut env, &b);
    let bufc = data::buffer_write(&mut env, &c);

    // Transfer data to GPU
    data::to_gpu(&mut env, &a, bufa);
    data::to_gpu(&mut env, &b, bufb);


    // Set kernel arguments
    kernel::setarg(&mut env, bufa); //a buffer
    kernel::setarg(&mut env, bufb); //b buffer
    kernel::setarg(&mut env, bufc); // c buffer

    // Run kernel with N threads
    kernel::run_kernel(&mut env, c.len());


    // Read results back from GPU
    data::from_gpu(&mut env, &mut c, bufc);

    println!("Result: {:?}", c);

    // Cleanup
    runtime::cleanvar(&mut env, bufa);
    runtime::cleanvar(&mut env, bufb);
    runtime::cleanvar(&mut env, bufc);
    runtime::cleanup(&mut env);
}

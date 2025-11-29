use obrah::data::*;
use obrah::kernel::*;
use obrah::runtime::*;
use std::fs::File;
use std::io;
use std::io::Read;
use std::io::Write;

fn save_ppm(
    path: &str,
    width: usize,
    height: usize,
    pixels: &[f32],
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Beginning file write...");
    let mut file = File::create(path)?; // make a new file
    // the ppm header is incredibly simple. just P6 for binary, then width height,
    // then max value of color/bit depth (65535 in this case, but usually 255), then pixel data. that's it!
    write!(file, "P6\n{} {}\n65535\n", width, height)?;
    // convert floats to u8
    for chunk in pixels.chunks_exact(4) {
        // we will get just the red, green, and blue values - no alpha in ppm.
        let r = (chunk[0].clamp(0.0, 1.0) * 65535.0) as u16;
        let g = (chunk[1].clamp(0.0, 1.0) * 65535.0) as u16;
        let b = (chunk[2].clamp(0.0, 1.0) * 65535.0) as u16;
        file.write_all(&r.to_be_bytes())?; // write red
        file.write_all(&g.to_be_bytes())?; // write green
        file.write_all(&b.to_be_bytes())?; // write blue
    }
    println!("Finished writing file!");
    Ok(())
}

fn read_input(inputmsg: &str) -> f32 {
    let mut input = String::new();
    println!("{inputmsg}");
    std::io::stdin()
        .read_line(&mut input)
        .expect("Failed reading input");

    let trimmed = input.trim();

    return match trimmed.parse::<f32>() {
        Ok(val) => val,
        Err(_) => panic!("Please enter a valid float."),
    };
}

fn read_texture(path: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    Ok(buffer)
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    const WIDTH: usize = 1000;
    const HEIGHT: usize = 1000;
    let mut env = Env::new(0, 0)?; // fix this with the right device - run example get_gpus to see all devices and platforms.
    env.use_kernel("examples/raytrace_kernel.cl")?
        .program()?
        .make_kernel("raytrace")?;

    let mut data = vec![0.0f32; WIDTH * HEIGHT * 4]; // we are going to make an empty buffer 1000 * 1000, and multiply by 4 for all 4 channels.
    let mut data_buf = Buffer::new(&mut env, &data);

    let mut input = String::new();

    println!("Custom settings? (y/N)");

    io::stdin()
        .read_line(&mut input)
        .expect("Failed reading input");

    input = input.trim().to_string();

    let mut sphere = [700.0f32, 500.0, 100.0, 100.0];
    let mut light = [500.0f32, 400.0, 300.0];

    if input.to_lowercase() == String::from("y") {
        let sphere_x = read_input("Enter sphere x: ");
        let sphere_y = read_input("Enter sphere y: ");
        let sphere_z = read_input("Enter sphere z: ");
        let sphere_rad = read_input("Enter sphere radius: ");
        sphere = [sphere_x, sphere_y, sphere_z, sphere_rad];
        let light_x = read_input("Enter light x: ");
        let light_y = read_input("Enter light y: ");
        let light_z = read_input("Enter light z: ");
        light = [light_x, light_y, light_z];
    }

    setarg(&env, &data_buf, 0)?; // our first argument is the empty buffer.
    setarg_scalar(&env, &WIDTH, 1); // the second argument is the width.
    setarg_scalar(&env, &HEIGHT, 2); // the third argument is the height.
    setarg_scalar(&env, &sphere, 3); // the fourth argument - THE SPHERE.
    setarg_scalar(&env, &light, 4); // the light is fifth

    let pixels = read_texture("texture.raw")?;
    let mut tex = Vec::with_capacity(pixels.len());
    for &b in &pixels {
        tex.push(b as f32 / 255.0);
    }
    let mut tex_buf = Buffer::new(&mut env, &tex);
    tex_buf.to(&mut env);
    setarg(&env, &tex_buf, 5)?; // sphere texture

    let pixels = read_texture("ground.raw")?;
    let mut tex = Vec::with_capacity(pixels.len());
    for &b in &pixels {
        tex.push(b as f32 / 255.0);
    }
    let mut tex_buf = Buffer::new(&mut env, &tex);
    tex_buf.to(&mut env);
    setarg(&env, &tex_buf, 6)?; // ground texture

    setarg_scalar(&env, &1, 7); // shadow the ground

    println!("Starting kernel execution...");
    run_kernel(&mut env, WIDTH, HEIGHT); // we are going to run the kernel at 1k by 1k resolution.
    println!("Kernel execution finished.");

    data_buf.from(&mut data, &mut env); // now, we are going to retrieve the data from the buffer.

    save_ppm("out.ppm", WIDTH, HEIGHT, &data).unwrap(); // all that is left is to save the ppm file.

    println!("Output saved.");
    Ok(())
}

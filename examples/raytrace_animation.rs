use obrah::data::*;
use obrah::kernel::*;
use obrah::runtime::*;
use std::fs::File;
use std::io::{Read, Write};

fn lerp(start: f32, end: f32, total: i32) -> Vec<f32> {
    // linear interpolation, very simple, just make a list of steps between one point and another
    let step = (end - start) / total as f32;
    let mut last = Vec::with_capacity((total + 1) as usize);
    for i in 0..=total {
        last.push(start + step * i as f32);
    }
    last
}

fn lerp_sphere(
    start_x: f32,
    end_x: f32,
    start_y: f32,
    end_y: f32,
    start_z: f32,
    end_z: f32,
    start_rad: f32,
    end_rad: f32,
    frames: i32,
) -> Vec<[f32; 4]> {
    // lerp an entire sphere
    let mut end = Vec::with_capacity(frames as usize);
    let all_x = lerp(start_x, end_x, frames);
    let all_y = lerp(start_y, end_y, frames);
    let all_z = lerp(start_z, end_z, frames);
    let all_rad = lerp(start_rad, end_rad, frames);
    for i in 0..frames {
        end.push([
            all_x[i as usize],
            all_y[i as usize],
            all_z[i as usize],
            all_rad[i as usize],
        ]);
    }
    end
}

fn bezier_sphere(
    // bezier curves are cool and i love them
    p1: [f32; 3],
    pc: [f32; 3],
    p2: [f32; 3],
    frames: usize,
    w_constant: f32,
) -> Vec<[f32; 4]> {
    let final_num_points = frames.max(2);
    let mut output_list = Vec::with_capacity(final_num_points);

    let step_size: f32 = 1.0 / (final_num_points as f32 - 1.0);

    for i in 0..final_num_points {
        let t = i as f32 * step_size;

        let one_minus_t = 1.0 - t;

        let weight1 = one_minus_t * one_minus_t;

        let weight_c = 2.0 * one_minus_t * t;

        let weight2 = t * t;

        let b_x = (p1[0] * weight1) + (pc[0] * weight_c) + (p2[0] * weight2);
        let b_y = (p1[1] * weight1) + (pc[1] * weight_c) + (p2[1] * weight2);
        let b_z = (p1[2] * weight1) + (pc[2] * weight_c) + (p2[2] * weight2);

        output_list.push([b_x, b_y, b_z, w_constant]);
    }

    output_list
}

fn read_texture(path: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    // just read a texture file and convert it into f32s because buffer
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    let mut tex = Vec::with_capacity(buffer.len());
    for &b in &buffer {
        tex.push(b as f32 / 255.0);
    }
    Ok(tex)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const WIDTH: usize = 1920;
    const HEIGHT: usize = 1080;
    let mut ffmpeg = std::process::Command::new("ffmpeg") // if you don't have ffmpeg... what are you doing with your life?
        .args([
            "-y", // overwrite output
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            &format!("{}x{}", WIDTH, HEIGHT),
            "-r",
            "30", // framerate
            "-i",
            "-", // read from stdin
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "output.mp4",
        ])
        .stdin(std::process::Stdio::piped())
        .spawn()?;

    let ffmpeg_stdin = ffmpeg.stdin.as_mut().unwrap();
    let mut env = Env::new(0, 0)?; // fix this with the right device - run example get_gpus to see all devices and platforms.
    env.use_kernel("examples/raytrace_kernel.cl")? // we want this kernel
        .program()? // program it
        .make_kernel("raytrace")?; // make sure it is the same name as in the kernel

    let mut data = vec![0.0f32; WIDTH * HEIGHT * 4]; // we are going to make an empty buffer 1920 * 1080, and multiply by 4 for all 4 channels of RGBA.
    let mut data_buf = Buffer::new(&mut env, &data);

    setarg(&env, &data_buf, 0)?; // our first argument is the empty buffer.

    setarg_scalar(&env, &WIDTH, 1); // the second argument is the width.
    setarg_scalar(&env, &HEIGHT, 2); // the third argument is the height.
    let tex = read_texture("texture.raw")?;
    let mut tex_buf = Buffer::new(&mut env, &tex);
    tex_buf.to(&mut env);
    setarg(&env, &tex_buf, 5)?; // sphere texture

    let tex = read_texture("ground.raw")?;
    let mut tex_buf = Buffer::new(&mut env, &tex);
    tex_buf.to(&mut env);
    setarg(&env, &tex_buf, 6)?; // ground texture (commented out for now, uncomment line 73 and comment line 74 in the raytracer if you want a texture)

    setarg_scalar(&env, &1, 7); // shadow the ground

    // now we are going to make an animation render thing
    let mut buf_frames: Vec<u8> = Vec::with_capacity(WIDTH * HEIGHT * 3 * 180); // poor ram. i apologise sincerely (not)
    let full_sphere = bezier_sphere(
        [1700.0, 200.0, 100.0],
        [0.0, 540.0, 100.0],
        [1700.0, 800.0, 100.0],
        180,
        100.0,
    );
    let start_light = lerp_sphere(1700.0, 600.0, 540.0, 540.0, 300.0, 300.0, 0.0, 0.0, 180);
    let mut full_light: Vec<[f32; 3]> = Vec::with_capacity(180);
    for i in start_light {
        full_light.push([i[0], i[1], i[2]]);
    }
    let time = std::time::Instant::now();
    for i in 1..=180 {
        let light = full_light[(i - 1) as usize];
        setarg_scalar(&env, &light, 4);
        let sphere = full_sphere[(i - 1) as usize];
        setarg_scalar(&env, &sphere, 3);

        run_kernel(&mut env, WIDTH, HEIGHT); // we are going to run the kernel at 1080p

        data_buf.from(&mut data, &mut env); // now, we are going to retrieve the data from the buffer.
        for chunk in data.chunks_exact(4) {
            // we will get just the red, green, and blue values - no alpha in raw.
            let r = (chunk[0].clamp(0.0, 1.0) * 255.0) as u8;
            let g = (chunk[1].clamp(0.0, 1.0) * 255.0) as u8;
            let b = (chunk[2].clamp(0.0, 1.0) * 255.0) as u8;
            buf_frames.extend_from_slice(&[r, g, b]);
        }

        println!("Saved frame {i}.");
    }

    ffmpeg_stdin.write_all(&buf_frames)?;
    ffmpeg_stdin.flush()?;
    drop(ffmpeg.stdin.take());

    ffmpeg.wait()?;

    println!("{:?} seconds to render video.", time.elapsed());

    println!("Output saved.");
    Ok(())
}

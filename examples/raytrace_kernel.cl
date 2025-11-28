__kernel void raytrace(__global float4 *image_buf, const int width,
                       const int height, const float4 sphere,
                       const float3 light) {

  int x = get_global_id(0); // x
  int y = get_global_id(1); // y
  if (x >= width ||
      y >= height) // this shouldn't happen, but if it does, ignore.
    return;

  int pixel = y * width + x; // calculate pixel offset

  float brightness = 0.0f;

  float3 rgb = (float3)(0.0f, 0.0f, 0.0f); // empty color

  float dx = x - sphere.x; // distance
  float dy = y - sphere.y;

  if ((dx * dx + dy * dy) <= sphere.w * sphere.w) { // solve the sphere equation
    float tz = sqrt(sphere.w * sphere.w - dx * dx -
                    dy * dy); // this gives us the z coordinate for a 3D sphere
    float z = sphere.z + tz;  // in global coordinates
    if (z >= 0) {             // we only want in front of camera
      float distl_x = light.x - x; // distance to light
      float distl_y = light.y - y;
      float distl_z = light.z - z;
      // onto brightness calculations
      float ldist = sqrt(distl_x * distl_x + distl_y * distl_y +
                         distl_z * distl_z); // formula for distance in 3D
      float I = 2.0f;                        // constant for inverse-square
      float3 surfacePos = (float3)(x, y, z); // the surface position
      float3 N = normalize(
          surfacePos -
          (float3)(sphere.x, sphere.y,
                   sphere.z)); // calculated by normalizing the difference
                               // between the sphere center and the point - this
                               // gives us a surface normal, or direction.
      float3 L = normalize((float3)(light.x, light.y, light.z) -
                           surfacePos); // the difference between the point and
                                        // the light gives us another thing
      float lambert = fmax(0.0f, dot(N, L)); // lambertian equation
      brightness =
          lambert *
          (I / (ldist * ldist) *
           16000.0f); // lambertian dot * inverse square is fairly accurate
      float3 baseColor = (float3)(70.0f / 255.0f, 130.0f / 255.0f,
                                  180.0f / 255.0f); // steel blue
      float ambient = 0.1f;                         // base brightness
      float3 V = (float3)(0.0f, 0.0f, 1.0f);
      float rim = pow(fmax(0.0f, 1.0f - dot(N, V)), 5.0f) * 0.25f;
      rgb = (ambient + brightness) * baseColor +
            rim * (float3)(0.3f, 0.5f, 1.0f); // multiply by brightness
    }
  }

  image_buf[pixel] =
      (float4)(rgb.x, rgb.y, rgb.z, 1.0f); // set the image buffer
}

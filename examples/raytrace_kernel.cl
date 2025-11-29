// literally the absolute simplest possible "raytracer" ever
// i wrote it in like 3 hours

__kernel void raytrace(__global float4 *image_buf, const int width,
                       const int height, const float4 sphere,
                       const float3 light, __global float4 *sphere_texture,
                       __global float4 *ground_texture,
                       const int shadowGround) {
  int x = get_global_id(0); // x
  int y = get_global_id(1); // y
  if (x >= width ||
      y >= height) // this shouldn't happen, but if it does, ignore.
    return;

  int pixel = y * width + x; // calculate pixel offset

  float brightness = 0.0f; // brightness starts off empty

  float3 rgb = (float3)(0.0f, 0.0f, 0.0f); // empty color

  float dx = x - sphere.x; // distance
  float dy = y - sphere.y;

  if ((dx * dx + dy * dy) <=
      sphere.w * sphere.w) { // solve the sphere equation - x^2 + y^2 should be
                             // less than r^2
    float tz = sqrt(sphere.w * sphere.w - dx * dx -
                    dy * dy); // this gives us the z coordinate for a 3D sphere
                              // - we just rearrange the equation.
    float z = sphere.z + tz;  // in global coordinates
    if (z >= 0) {             // we only want in front of camera
      float distl_x = light.x - x; // distance to light
      float distl_y = light.y - y;
      float distl_z = light.z - z;
      // onto brightness calculations
      float ldist =
          sqrt(distl_x * distl_x + distl_y * distl_y +
               distl_z * distl_z); // formula for euclidean distance in 3D
      float I = 2.0f;              // constant for inverse-square
      float3 surfacePos = (float3)(x, y, z); // the surface position
      // i will now explain lambertian shading
      // N is our surface normal. basically, draw an arrow from
      // the center to the point - that arrow's direction
      // is the normal, or where the surface is pointing.
      // L is the direction from the point to the light.
      // a dot product tells us how similar 2 directions are -
      // if the number is positive, they point in roughly the same direction.
      // if it is 0, they are perpendicular.
      // and if it is negative they point in roughly opposite
      // directions.
      // this works because geometrically, a dot product
      // is just the length of a * the length of b * the cosine of
      // the angle between them.
      // if the angle between them is 0 (they point in the same direction),
      // cos(0) = 1 if the angle between them is 90 (they are perpendicular),
      // cos(90) = 0
      // and finally, if the angle between them is 180 (they are in
      // exact opposite directions), cos(180) = -1
      // i hope that makes sense
      // anyway
      // how does this relate to our lambertian dot?
      // well, if we dot N and L, we basically get how much
      // the surface is pointing in the direction of the light.
      // if it is hitting head on, you'd expect it to be higher.
      // very simple but adds much more realism
      // why not try commenting it out to see the difference yourself?
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
      float ambient = 0.1f; // base brightness
      float3 V = (float3)(0.0f, 0.0f, 1.0f);

      float rim = pow(fmax(0.0f, 1.0f - dot(N, V)), 5.0f) *
                  0.25f; // i sincerely have no idea why this works but it does
      // texture time!
      float u =
          0.5f + atan2(N.z, N.x) / M_2_PI_F; // u is our horizontal component
      float v = 0.5f - asin(N.y) / M_PI_F;   // v is our vertical component
      int tx = (int)(u * 2048); // we want them relative to our textures.
      int ty = (int)(v * 1024);
      tx = clamp(tx, 0, 2047);
      ty = clamp(ty, 0, 1023);
      int idx = ty * 2048 + tx;
      float4 tex = sphere_texture[idx];
      float3 baseColor = tex.xyz;
      rgb = (ambient + brightness) * baseColor +
            rim * (float3)(0.3f, 0.5f,
                           1.0f); // multiply by ambient and diffuse
                                  // (brightness and lambert)
    }
  } else { // ground (top down)
    float z = 0.0f;
    int idx = y * 2048 + x;
    float4 tex = ground_texture[idx];
    // float3 baseColor = tex.xyz;
    float3 baseColor = (float3)(1.0f, 0.0f, 0.0f);
    float3 ambient = 0.1f;
    float I = 2.0f;
    float distl_x = light.x - x; // distance to light, same as regular
    float distl_y = light.y - y;
    float distl_z = light.z - z;
    // complex math time!
    float shadowMult = 1.0f;
    // math: say we have triangle PLC, where P is the ground point, L is the
    // light, and C is the circle. if we get the perpendicular of C to line PL,
    // and we mark the point they contact as X, we now know that the height of
    // this triangle - the distance from X to C - is the closest point from the
    // line to the sphere. if this point is less than the radius, its inside! if
    // not, its outside. to get X, we do P + t(L - P) - super simple. the
    // formula says that the point X is the point P, plus how much along the
    // direction of the line PL we have to travel. t is how much we have to
    // travel - if t = 1, we have to travel the whole length of the line. if t =
    // 0, we dont have to travel at all. but how do we get t? using some complex
    // math i will now explain.
    float3 v =
        (float3)(sphere.x, sphere.y, sphere.z) -
        (float3)(x, y, z); // this is our vector (direction) to the sphere
    float3 w = light - (float3)(x, y, z); // this is our vector to the light.
    // imagine we draw a perpendicular line to PL that hits C. where that line
    // crosses is called X, let's say. now we have a right angled
    // triangle with C, P (our point) and X. we want to figure out t - t is how
    // far along we have to go. since t is how far along we go, that means it is
    // the line from the point P to X, which is the adjacent of our triangle. we
    // know the hypotenuse - it is simply |v| (remember, |vector| just means the
    // distance between the 2 points the vector refers to), the line from the
    // point to the sphere. since cos(angle) = adj / hypo, we can substitute the
    // values. cos(angle) = |w| / |v|. if we rearrange it? |w| = |v| *
    // cos(angle). how does this relate to dot products? well, the dot product
    // of v and w is |v| * |w| * cos(angle). wait! we have |v| *
    // cos(angle) in there! if we rearrange it, we get |v| * cos(angle) / |w|,
    // or dot(v, w) / |w| - but we don't want the full length; rather, we want
    // one between 0 and 1, or a percentage. to do this, we just divide by |w|,
    // which is the max possible distance. this brings our final equation to
    // dot(v, w) / |w|^2. if you look down below, that is exactly what we have.
    float dist =
        sqrt((light.x - x) * (light.x - x) + (light.y - y) * (light.y - y) +
             (light.z - z) * (light.z - z));
    float t =
        dot(v /* vector of sphere & point*/, w /* vector of light and point*/) /
        (dist * dist);
    // this makes it super easy; if you remember our earlier formula,
    // P + t(L - P),
    // you can see that t basically just means how far along the
    // line we travel.
    // if t is more than 1, the sphere is behind the light, and cannot cast a
    // shadow on us.
    // if it is less than 0, it is behind us, and cannot cast a shadow on us.
    float3 sphere2 = (float3)(sphere.x, sphere.y, sphere.z);
    if (0 <= t && t <= 1) {
      // now the formula for X
      float3 X = (float3)(x, y, z) + t * (w); // calculating X becomes trivial.
      // and since we know X is the closest point to the sphere center,
      // we just check if it is less than the radius. if it is?
      // then the line intersects the circle.
      float h = sqrt((X.x - sphere2.x) * (X.x - sphere2.x) +
                     (X.y - sphere2.y) * (X.y - sphere2.y) +
                     (X.z - sphere2.z) * (X.z - sphere2.z));
      if (h < sphere.w) {
        shadowMult = 0.3f;
      }
    }
    // onto brightness calculations
    float3 surfacePos = (float3)(x, y, z); // the surface position
    float3 N = (float3)(0.0f, 0.0f, 1.0f);
    float3 L = normalize((float3)(light.x, light.y, light.z) -
                         surfacePos); // the difference between the point and
                                      // the light gives us another thing
    float lambert = fmax(0.0f, dot(N, L)); // lambertian equation
    float ldist = sqrt(distl_x * distl_x + distl_y * distl_y +
                       distl_z * distl_z); // euclidean formula for distance
    brightness =
        lambert * (I / (ldist * ldist) * 16000.0f) *
        shadowMult; // put together our lambertian, inverse-square, and shadow
    if (shadowGround) { // sometimes you might want a black, shadowless ground
      rgb = (ambient + brightness) * baseColor;
    } else {
      rgb = baseColor;
    }
  }

  image_buf[pixel] =
      (float4)(rgb.x, rgb.y, rgb.z, 1.0f); // set the image buffer to rgb.
}

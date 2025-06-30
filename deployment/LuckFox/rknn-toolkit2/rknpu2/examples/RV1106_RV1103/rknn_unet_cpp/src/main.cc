#include "rknn_api.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb/stb_image_resize.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"


static void* load_file(const char *path, size_t *size_out) {
  FILE *fp = fopen(path, "rb");
  if (!fp) return NULL;
  fseek(fp, 0, SEEK_END);
  size_t size = ftell(fp);
  rewind(fp);
  void *data = malloc(size);
  fread(data, 1, size, fp);
  fclose(fp);
  *size_out = size;
  return data;
}

static unsigned char* load_and_resize_image(const char *img_path, int target_h, int target_w, int channels) {
  int w, h, c;
  unsigned char *input_img = stbi_load(img_path, &w, &h, &c, channels);
  if (!input_img) {
    printf("Failed to load image: %s\n", img_path);
    return NULL;
  }

  unsigned char *resized = (unsigned char*)malloc(target_h * target_w * channels);
  stbir_resize_uint8(input_img, w, h, 0, resized, target_w, target_h, 0, channels);
  stbi_image_free(input_img);
  printf("Loaded image %s (%d x %d x %d), resized to %d x %d x %d\n", img_path, w, h, c, target_w, target_h, channels);
  return resized;
}

static inline int64_t get_time_us() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec * 1000000 + tv.tv_usec;
}

void save_mask_as_png(const char* filename, const uint8_t* mask, int width, int height) {
  const uint8_t class_colors[6][3] = {
      { 0,   0,   0   }, // class 0: black
      { 0,   255, 0   }, // class 1: green
      { 0,   0,   255 }, // class 2: blue
      { 255, 0,   0   }, // class 3: red
      { 255, 255, 0   }, // class 4: yellow
      { 255, 0,   255 }  // class 5: magenta
  };

  uint8_t* rgb_image = (uint8_t*)malloc(width * height * 3);
  if (!rgb_image) {
    printf("Failed to allocate memory for PNG\n");
    return;
  }

  for (int i = 0; i < width * height; ++i) {
    uint8_t class_id = mask[i];
    if (class_id >= 6) class_id = 0; // fallback
    rgb_image[i * 3 + 0] = class_colors[class_id][0];
    rgb_image[i * 3 + 1] = class_colors[class_id][1];
    rgb_image[i * 3 + 2] = class_colors[class_id][2];
  }

  if (stbi_write_png(filename, width, height, 3, rgb_image, width * 3)) {
    printf("Saved colored mask: %s\n", filename);
  } else {
    printf("Failed to save PNG: %s\n", filename);
  }

  free(rgb_image);
}


int main(int argc, char **argv) {
  if (argc < 3) {
    printf("Usage: %s <model.rknn> <input.jpg>\n", argv[0]);
    return -1;
  }

  const char *model_path = argv[1];
  const char *image_path = argv[2];

  int64_t start;
  int64_t end;

  start = get_time_us();
  // Load model
  size_t model_size;
  void *model_data = load_file(model_path, &model_size);
  if (!model_data) {
    printf("Failed to load model file.\n");
    return -1;
  }

  rknn_context ctx;
  int ret = rknn_init(&ctx, model_data, model_size, 0, NULL);
  free(model_data);
  if (ret != RKNN_SUCC) {
    printf("rknn_init failed: %d\n", ret);
    return -1;
  }

  // Get input shape
  rknn_tensor_attr input_attr;
  memset(&input_attr, 0, sizeof(input_attr));
  input_attr.index = 0;
  rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &input_attr, sizeof(input_attr));
  int input_h = input_attr.dims[1];
  int input_w = input_attr.dims[2];
  int input_c = input_attr.dims[3];
  printf("Model input: %d x %d x %d\n", input_h, input_w, input_c);

  // Load image
  unsigned char* input_data = load_and_resize_image(image_path, input_h, input_w, input_c);
  if (!input_data) return -1;

  // Allocate and bind input
  rknn_tensor_mem* input_mem = rknn_create_mem(ctx, input_h * input_w * input_c);
  memcpy(input_mem->virt_addr, input_data, input_h * input_w * input_c);
  free(input_data);
  ret = rknn_set_io_mem(ctx, input_mem, &input_attr);
  if (ret != RKNN_SUCC) {
    printf("rknn_set_io_mem failed: %d\n", ret);
    return -1;
  }

  // Get output attr
  rknn_tensor_attr output_attr;
  memset(&output_attr, 0, sizeof(output_attr));
  output_attr.index = 0;
  rknn_query(ctx, RKNN_QUERY_NATIVE_NHWC_OUTPUT_ATTR, &output_attr, sizeof(output_attr));
  int out_h = output_attr.dims[1];
  int out_w = output_attr.dims[2];
  int out_c = output_attr.dims[3];
  printf("Model output: %d x %d x %d\n", out_h, out_w, out_c);

  // Allocate and bind output
  rknn_tensor_mem* output_mem = rknn_create_mem(ctx, out_h * out_w * out_c);
  ret = rknn_set_io_mem(ctx, output_mem, &output_attr);
  if (ret != RKNN_SUCC) {
    printf("rknn_set_io_mem (output) failed: %d\n", ret);
    return -1;
  }
  end = get_time_us();
  float preproc_time = (end - start) / 1000.0f;
  
  // Run inference
  printf("Running inference...\n");
  start = get_time_us();
  ret = rknn_run(ctx, nullptr);
  end = get_time_us();
  if (ret != RKNN_SUCC) {
    printf("rknn_run failed: %d\n", ret);
    return -1;
  }
  float infer_time = (end - start) / 1000.0f;

  // Get output
  start = get_time_us();
  uint8_t* output_data = (uint8_t*)output_mem->virt_addr;

  // Postprocess — argmax over channels
  int out_size = out_h * out_w;
  uint8_t* seg_mask = (uint8_t*)malloc(out_size);

  for (int h = 0; h < out_h; ++h) {
    for (int w = 0; w < out_w; ++w) {
      int base = (h * out_w + w) * out_c;
      int max_class = 0;
      uint8_t max_val = output_data[base];
      for (int c = 1; c < out_c; ++c) {
        uint8_t val = output_data[base + c];
        if (val > max_val) {
          max_val = val;
          max_class = c;
        }
      }
      seg_mask[h * out_w + w] = max_class;
    }
  }

  // Save output
  save_mask_as_png("seg_mask.png", seg_mask, out_w, out_h);
  end = get_time_us();

  float postproc_time = (end - start) / 1000.0f;
  printf("Preprocess time: %.2f ms\n", preproc_time);
  printf("Inference time: %.2f ms\n", infer_time);
  printf("Postprocess time: %.2f ms\n", postproc_time);
  
  // Cleanup
  free(seg_mask);
  rknn_destroy_mem(ctx, input_mem);
  rknn_destroy_mem(ctx, output_mem);
  rknn_destroy(ctx);

  return 0;
}

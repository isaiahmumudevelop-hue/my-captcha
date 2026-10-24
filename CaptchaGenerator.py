from CaptchaGenerator import CaptchaConfig, ImageGridCaptcha

# 1. Configure the image size and grid
config = CaptchaConfig(width=720, height=720, random_seed=42)

# 2. Generate the challenge
result = ImageGridCaptcha(config).generate(
    path_folder="./images",      # Your folder of categorized images
    target_name="traffic_light", # The object users must select
    name_export="image_grid",
    path_export="./output",      # Where to save the final image
    grid_size=3,                 # 3x3 grid
)

# 3. Inspect the results
print("Image saved to:", result.path)
print("Instruction for user:", result.prompt)
print("Correct answer (keep server-side!):", result.answer) 
# result.answer will be a list of correct cell indices, e.g., [0, 1, 3, 4]

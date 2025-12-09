def generate_random_images(size=1024, image_numbers=30):
    random_images = []
    for i in range(image_numbers):
        random_sample = np.random.random((size, size))
        random_images.append(random_sample)
    return random_images
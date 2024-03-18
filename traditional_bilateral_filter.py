import cv2
import os
import csv
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function - apply bilateral filter to an image
def apply_bilateral_filter(image_path, d, sigmaColor, sigmaSpace):
    image = cv2.imread(image_path)
    filtered_image = cv2.bilateralFilter(image, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)
    return filtered_image

# Function - calculate similarity between two images
def calculate_similarity(image1, image2):
    new_image = cv2.absdiff(image1, image2)
    similarity = 1 - (new_image.sum() / (image1.shape[0] * image1.shape[1] * image1.shape[2]))
    return similarity

# Function - process images and calculate similarities
def process_images(dataset_path, original_folder):
    # Generate parameter sets
    d_values = range(3, 10,2)  # d values from 3 (as mimimum kernal size) to 9 in steps of two for even kernel
    sigma_values = range(0, 300,20)  # sigmaColor and sigmaSpace values from 1 to 300

    total_iterations = len(d_values) * len(sigma_values) * len(sigma_values)
    current_iteration = 0

    # Initialize dictionary to store average similarities for each parameter 
    all_average_similarities = {}
    # Iterate through each parameter set
    for d in d_values:
        for sigmaColor in sigma_values:
            for sigmaSpace in sigma_values:

                current_iteration += 1
                completion_percentage = (current_iteration / total_iterations) * 100

                # For each parameter set, process images and calculate similarities
                print(f"Processing with parameters: d={d}, sigmaColor={sigmaColor}, sigmaSpace={sigmaSpace} - {completion_percentage:.2f}% complete")
                similarities = []  # Initialize an empty list for each set of parameters
                noise_levels = ['noisy5', 'noisy10', 'noisy15', 'noisy25', 'noisy35', 'noisy50']
                # Iterate through each noise level
                for noise_level in noise_levels:
                    path = os.path.join(dataset_path, noise_level)
                    for img_filename in os.listdir(path):
                        noisy_image_path = os.path.join(path, img_filename)
                        original_image_path = os.path.join(dataset_path, original_folder, img_filename)

                        # Check if the original image exists
                        if os.path.exists(original_image_path):
                            filtered_image = apply_bilateral_filter(noisy_image_path, d, sigmaColor, sigmaSpace)
                            original_image = cv2.imread(original_image_path)
                            similarity = calculate_similarity(filtered_image, original_image)
                            similarities.append((similarity, img_filename, noise_level))
                        else:
                            print(f"Original image for {img_filename} not found.")
                # Ensure there are similarities calculated before trying to compute average
                if similarities: 
                    avg_similarity = sum([item[0] for item in similarities]) / len(similarities)
                    print(f"Average similarity for parameters (d={d}, sigmaColor={sigmaColor}, sigmaSpace={sigmaSpace}): {avg_similarity:.2f}")
                    all_average_similarities[(d, sigmaColor, sigmaSpace)] = avg_similarity

    return all_average_similarities

# Function - graph the average similarities
def graph(average_similarities):
    # Extract the parameters and average similarities
    d_values = []
    sigmaColor_values = []
    sigmaSpace_values = []
    avg_similarities = []

    for params, avg_similarity in average_similarities.items():
        d_values.append(params[0])
        sigmaColor_values.append(params[1])
        sigmaSpace_values.append(params[2])
        avg_similarities.append(avg_similarity)

    # Create a 4D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the data points
    img = ax.scatter(sigmaSpace_values, sigmaColor_values, avg_similarities, c=d_values, cmap=plt.hot())
    fig.colorbar(img, label='d')

    # Set labels and title
    ax.set_xlabel('Sigma Space')
    ax.set_ylabel('Sigma Color')
    ax.set_zlabel('Average Similarity')
    ax.set_title('Average Similarities')

    # Show the plot
    plt.show()

# Function - load average similarities from a CSV file
def load_average_similarities_from_csv():
    csv_files = [file for file in os.listdir() if file.endswith('.csv')]
    for i, file in enumerate(csv_files):
        print(f"{i+1}. {file}")

    selected_csv = int(input("Enter the number of the CSV file you want to load: ")) - 1
    selected_csv_file = csv_files[selected_csv]

    average_similarities = {}
    with open(selected_csv_file, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            d = int(row[0])
            sigmaColor = int(row[1])
            sigmaSpace = int(row[2])
            avg_similarity = float(row[3])
            average_similarities[(d, sigmaColor, sigmaSpace)] = avg_similarity
    return average_similarities

# Function - save average similarities to a CSV file
def save_average_similarities_to_csv(average_similarities):
    csv_file = "average_similarities_1.csv"
    count = 1
    while os.path.exists(csv_file):
        count += 1
        csv_file = csv_file.replace(f'_{count-1}.csv', f'_{count}.csv')

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writeroyw(["d", "sigmaColor", "sigmaSpace", "average_similarity"])
        for params, avg_similarity in average_similarities.items():
            writer.writerow([params[0], params[1], params[2], avg_similarity])

    print("Average similarities saved to CSV file.")

if __name__ == "__main__":
    dataset_path = "CBSD68"
    original_folder = "original_png"

    choice = input("Do you want to load average similarities from a CSV file? (y/n): ")
    if choice.lower() == "y":
        average_similarities = load_average_similarities_from_csv()
        graph(average_similarities)
    else:
        average_similarities = process_images(dataset_path, original_folder)
        save_average_similarities_to_csv(average_similarities)


    
    
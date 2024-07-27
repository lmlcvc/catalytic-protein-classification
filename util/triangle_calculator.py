import numpy as np


# Function to calculate the mean and standard deviation of side c
def calculate_side_c(mu_a, sigma_a, mu_b, sigma_b, mu_alpha, sigma_alpha, mu_beta, sigma_beta):
    # Convert angles from degrees to radians
    mu_alpha_rad = np.radians(mu_alpha)
    mu_beta_rad = np.radians(mu_beta)
    sigma_alpha_rad = np.radians(sigma_alpha)
    sigma_beta_rad = np.radians(sigma_beta)

    # Calculate gamma
    mu_gamma = 180 - (mu_alpha + mu_beta)
    mu_gamma_rad = np.radians(mu_gamma)
    sigma_gamma_rad = np.sqrt(sigma_alpha_rad ** 2 + sigma_beta_rad ** 2)

    # Calculate the mean of c using the Law of Cosines
    mu_c = np.sqrt(mu_a ** 2 + mu_b ** 2 - 2 * mu_a * mu_b * np.cos(mu_gamma_rad))

    # Calculate the partial derivatives
    partial_a = (mu_a - mu_b * np.cos(mu_gamma_rad)) / mu_c
    partial_b = (mu_b - mu_a * np.cos(mu_gamma_rad)) / mu_c
    partial_gamma = (mu_a * mu_b * np.sin(mu_gamma_rad)) / mu_c

    # Calculate the standard deviation of c
    sigma_c = np.sqrt((partial_a * sigma_a) ** 2 + (partial_b * sigma_b) ** 2 + (partial_gamma * sigma_gamma_rad) ** 2)

    return mu_c, sigma_c


# Example values
mu_a = 3.05
sigma_a = 0.45
mu_b = 2.89
sigma_b = 0.57
mu_alpha = 18.63  # degrees
sigma_alpha = 5.06  # degrees
mu_beta = 131.85  # degrees
sigma_beta = 15.89  # degrees

# Calculate the mean and standard deviation of side c
mu_c, sigma_c = calculate_side_c(mu_a, sigma_a, mu_b, sigma_b, mu_alpha, sigma_alpha, mu_beta, sigma_beta)

print(f"Mean of side c: {mu_c}")
print(f"Standard deviation of side c: {sigma_c}")

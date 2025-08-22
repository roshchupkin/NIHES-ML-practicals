import pandas as pd
import numpy as np


### CODE to generate a dataset for Data Science Course 2023 ###

def generate_corrected_enhanced_data_v3(num_samples=10000):
    np.random.seed(0)  # Set seed for reproducibility
    
    # Generate basic features
    ages = np.random.randint(20, 80, num_samples)
    sexes = np.random.choice(['M', 'F'], num_samples)
    
    # Different average heights for males and females
    male_heights = np.random.normal(175, 7, num_samples)
    female_heights = np.random.normal(162, 6, num_samples)
    heights = np.where(sexes == 'M', male_heights, female_heights)
    
    weights = np.random.normal(70, 15, num_samples)
    
    # BMI = weight(kg) / height(m)^2
    bmi = weights / (heights/100)**2
    
    # Generate health features
    cholesterol = np.random.normal(180 + ages * 0.2, 25, num_samples)
    blood_pressure = np.random.normal(75 + ages * 0.5, 10, num_samples)
    calorie_intake = np.random.normal(2200 - ages * 2, 300, num_samples)
    exercise_frequency = np.random.randint(0, 7, num_samples)  # 0-7 days a week
    
    # Generate diabetes outcomes based on certain conditions
    diabetes = np.zeros(num_samples)
    diabetes_conditions = [
        (ages > 50) & (cholesterol > 220),
        (bmi > 30) & (exercise_frequency < 3),
        (blood_pressure > 90) & (exercise_frequency < 2)
    ]
    
    # Setting diabetes labels based on conditions and gender prevalence
    for condition in diabetes_conditions:
        male_condition_indices = np.where(np.logical_and(condition, sexes == 'M'))[0]
        female_condition_indices = np.where(np.logical_and(condition, sexes == 'F'))[0]
        
        male_diabetes_indices = male_condition_indices[np.random.rand(len(male_condition_indices)) < 0.15]
        female_diabetes_indices = female_condition_indices[np.random.rand(len(female_condition_indices)) < 0.10]
        
        diabetes[male_diabetes_indices] = 1
        diabetes[female_diabetes_indices] = 1

    # Add noise to diabetes labels
    noise_indices = np.random.choice(num_samples, size=int(0.1 * num_samples), replace=False)
    diabetes[noise_indices] = 1 - diabetes[noise_indices]
    
    # Introduce some missing data
    cholesterol[np.random.choice(num_samples, 100, replace=False)] = np.nan
    blood_pressure[np.random.choice(num_samples, 100, replace=False)] = np.nan
    
    # Create the dataframe first
    df = pd.DataFrame({
        'subject_id': range(1, num_samples+1),
        'age': ages,
        'sex': sexes,
        'weight': weights,
        'height': heights,
        'bmi': bmi,
        'cholesterol': cholesterol,
        'blood_pressure': blood_pressure,
        'calorie_intake': calorie_intake,
        'exercise_frequency': exercise_frequency,
        'diabetes': diabetes
    })
    
    # Introduce extreme outliers
    outlier_indices = np.random.choice(num_samples, 10, replace=False)
    features = ['age', 'weight', 'height', 'cholesterol', 'blood_pressure', 'calorie_intake', 'exercise_frequency', 'bmi']
    for i, feature in enumerate(features):
        extreme_value = 5 * df[feature].std() + df[feature].mean()  # 5 standard deviations from the mean
        df.at[outlier_indices[i], feature] = extreme_value
    
    # Add temporal data (date of medical checkup)
    start_date = pd.to_datetime('2020-01-01')
    end_date = pd.to_datetime('2022-01-01')
    date_range = pd.date_range(start_date, end_date, freq='D').to_numpy()
    checkup_dates = np.random.choice(date_range, num_samples)
    df['checkup_date'] = checkup_dates
    
    return df

# Generate the enhanced dataset with corrections
df_corrected_v3 = generate_corrected_enhanced_data_v3()
df_corrected_v3.head()


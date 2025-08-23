# Supervised Learning Practicals - NIHES Data Science Course

This repository contains comprehensive practical materials for learning supervised machine learning techniques, specifically focused on diabetes prediction using real-world health data.

## 📚 Course Overview

These practicals are designed for the **NIHES (Netherlands Institute for Health Sciences) Data Science for Epidemiology Summer Course 2025**. The materials provide hands-on experience with:

- **Data Quality Assessment and Cleaning**
- **Exploratory Data Analysis (EDA)**
- **Supervised Learning Model Development**
- **Cross-Validation Techniques**
- **Hyperparameter Tuning**
- **Feature Engineering and Selection**

## 🎯 Learning Objectives

By completing these practicals, you will be able to:

1. **Understand the complete ML workflow** - from data exploration to model deployment
2. **Apply robust evaluation techniques** - including cross-validation and proper train/test splits
3. **Compare multiple algorithms** - Decision Trees, Random Forests, SVM, and more
4. **Handle real-world data challenges** - outliers, missing values, class imbalance
5. **Interpret model results** - using appropriate metrics for different scenarios

## 📁 Repository Structure

```
Practical_I/
├── README.md                                    # This file
├── dataset_24082023.csv                        # Original dataset
├── data_cleaned.csv                            # Cleaned dataset (output from Practical 1)
├── dataset_generator.py                        # Python script to generate synthetic data
├── Supervised_learning_2025.Rmd               # Practical 1: Data QC & Basic Modeling
├── Supervised_learning_2025.html              # HTML output of Practical 1
├── Supervised_learning_2025.pdf               # PDF output of Practical 1
├── Supervised_Learning_Practical_2_CV.Rmd     # Practical 2: Cross-Validation
├── Supervised_Learning_Practical_2_CV.html    # HTML output of Practical 2
├── Supervised_Learning_Advanced_Practicals.R  # Advanced techniques (R script)
└── Supervised_learning_2025_files/            # Supporting files for Practical 1
    └── figure-html/
└── Supervised_Learning_Practical_2_CV_files/  # Supporting files for Practical 2
    └── figure-html/
```

## 🏥 Dataset Description

The practicals use a synthetic health dataset with the following features:

- **Demographics**: Age, Sex, Weight, Height, BMI
- **Health Metrics**: Cholesterol, Blood Pressure, Calorie Intake, Exercise Frequency
- **Outcome Variable**: Diabetes Status (Binary: 0/1)
- **Temporal Data**: Medical Checkup Dates

The dataset includes realistic challenges such as:
- Missing values
- Outliers
- Class imbalance
- Correlated features

## 📖 Practical Sessions

### Practical 1: Data Quality Check, Exploration, and Basic Modeling
**File**: `Supervised_learning_2025.Rmd`

**Topics Covered**:
- Data loading and initial exploration
- Outlier detection using boxplots and Z-scores
- Missing value handling
- Feature correlation analysis
- Basic model training (Decision Trees, Random Forest, SVM)
- Model evaluation with accuracy metrics

**Key Skills**:
- Data preprocessing techniques
- Exploratory data analysis
- Basic supervised learning implementation

### Practical 2: Cross-Validation and Model Comparison
**File**: `Supervised_Learning_Practical_2_CV.Rmd`

**Topics Covered**:
- Limitations of single train/test splits
- K-fold cross-validation implementation
- Multiple model comparison
- ROC-AUC evaluation for imbalanced data
- Model selection strategies

**Key Skills**:
- Robust model evaluation
- Cross-validation techniques
- Model comparison methodologies

### Advanced Practical: Hyperparameter Tuning and Feature Engineering
**File**: `Supervised_Learning_Advanced_Practicals.R`

**Topics Covered**:
- Hyperparameter optimization
- Feature selection techniques
- Advanced model evaluation
- Ensemble methods
- Model interpretation

**Key Skills**:
- Model optimization
- Feature engineering
- Advanced evaluation metrics

## 🛠️ Technical Requirements

### R Packages Required
```r
# Core packages
tidyverse      # Data manipulation and visualization
caret          # Machine learning workflow
rpart          # Decision trees
randomForest   # Random forests
e1071          # SVM models
corrplot       # Correlation visualization
GGally         # Pair plots
rpart.plot     # Tree visualization
pROC           # ROC curves
xgboost        # Gradient boosting
```

### Installation
```r
# Install required packages
required_packages <- c("tidyverse", "caret", "rpart", "randomForest", 
                      "e1071", "corrplot", "GGally", "rpart.plot", 
                      "pROC", "xgboost")

# Install missing packages
missing_packages <- required_packages[!(required_packages %in% installed.packages()[,"Package"])]
if(length(missing_packages) > 0) {
  install.packages(missing_packages)
}
```

### Python Requirements (for data generation)
```bash
pip install pandas numpy
```

## 🚀 Getting Started

1. **Clone the repository**:
   ```bash
   git clone [your-repository-url]
   cd NIHES-ML-practicals
   ```

2. **Install R packages** (run in R console):
   ```r
   source("setup_packages.R")  # If available, or run the installation code above
   ```

3. **Start with Practical 1**:
   - Open `Supervised_learning_2025.Rmd` in RStudio
   - Run the code chunks sequentially
   - Complete the exercises marked with `# TODO:`

4. **Progress through the practicals**:
   - Complete Practical 1 before moving to Practical 2
   - Use the HTML/PDF outputs as reference materials
   - Experiment with the advanced techniques in the R script

## 📊 Expected Outcomes

After completing these practicals, you will have:

- **Hands-on experience** with the complete ML workflow
- **Understanding** of data preprocessing importance
- **Skills** to evaluate and compare different algorithms
- **Knowledge** of cross-validation and model selection
- **Ability** to handle real-world data challenges

## 🤝 Contributing

This repository is designed for educational purposes. If you find errors or have suggestions for improvements, please:

1. Create an issue describing the problem
2. Fork the repository and submit a pull request
3. Ensure all code follows R best practices

## 📄 License

This educational material is provided for academic use. Please respect the educational nature of this content.

## 👥 Authors

- **NIHES Teaching Team** - *Initial work* - [NIHES](https://www.nihes.com/)
- **Course Instructors** - *Practical Development*

## 🙏 Acknowledgments

- Erasmus MC for hosting the course
- NIHES for educational framework
- R community for excellent packages
- Students for feedback and testing

---

**Note**: This repository contains educational materials. The dataset is synthetic and created for learning purposes. Always ensure proper data handling and privacy considerations when working with real health data.

## 📞 Support

For questions about the practicals, please contact the course instructors or create an issue in this repository.

---

*Last updated: January 2025*

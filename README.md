# FlashDroughts

A Python package for detecting, analyzing, and visualizing flash drought events from climate data.

## Features
- **Data Handling:** Weekly climate data (precipitation, PET, VWC), rolling sums, seasonal subsets.  
- **Drought Detection:** Onset detection via z-score thresholds or regression residuals; customizable criteria.  
- **Statistical Analysis:** KDE histograms, Pearson Type III fitting, regression diagnostics.  
- **Visualization:** Plots for onset distributions, residual trends, and seasonal patterns.

## Installation

Clone the repository and install locally:

```bash
git clone https://github.com/yourusername/FlashDroughts.git
cd FlashDroughts
pip install -e .
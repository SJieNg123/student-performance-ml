# Raw Input Integration - Quick Guide

## Overview

The Streamlit dashboard now accepts **human-readable raw inputs** instead of preprocessed values. Users can enter values like "High", "Male", "Yes" instead of encoded numbers.

## What Changed

### New File: `preprocessor.py`

This module handles the transformation from raw inputs to preprocessed format:

- **RawInputPreprocessor**: Main class that applies ordinal encoding and sklearn preprocessing
- **RAW_INPUT_CONFIG**: Dictionary defining all input features with their types and validation rules
- **Validation**: Ensures inputs are within valid ranges

### Updated: `streamlit_app.py`

The prediction page now features:

1. **Organized Input Form** with 6 sections:
   - 📚 Academic Information
   - 🎓 Support & Resources
   - 🏫 School Environment
   - 👤 Personal Factors  
   - 📊 Demographics & Other
   - Additional Factors

2. **Human-Readable Inputs**:
   - **Numeric fields**: Number inputs with min/max validation (e.g., Hours Studied: 0-44)
   - **Categorical fields**: Dropdowns with options (e.g., Gender: Male/Female)
   - **Ordinal fields**: Dropdowns with ordered options (e.g., Parental Involvement: None/Low/Medium/High)

3. **Student Profile Analysis**:
   - **Strong Points**: Highlights positive factors (e.g., "✅ Excellent study hours")
   - **Areas for Improvement**: Suggests improvements (e.g., "⚠️ Increase study hours")

## How It Works

### Preprocessing Pipeline

```
Raw Input (User enters: "High", "Male", 25)
    ↓
Ordinal Encoding (High → 3, Male → stays as-is)
    ↓
One-Hot Encoding (Male → [1, 0] or [0, 1] depending on drop_first)
    ↓
Standardization (Hours Studied: 25 → (25 - mean) / std)
    ↓
Model-Ready Input (numpy array)
```

### Code Flow

1. User fills form in Streamlit → `input_values` dictionary created
2. `processor.validate_input()` → Checks all values are valid
3. `processor.transform_raw_input()` → Applies preprocessing
4. `ensemble.predict_all()` → Makes predictions
5. Results displayed with analysis

## Example Usage

### Running the Dashboard

```bash
streamlit run streamlit_app.py
```

### Sample Input

```python
{
    'Hours_Studied': 25,
    'Attendance': 90,
    'Parental_Involvement': 'High',  # Not encoded!
    'Access_to_Resources': 'Medium',
    'Gender': 'Male',
    # ... other features
}
```

This gets automatically converted to the preprocessed format the models expect.

## Feature Mappings

### Ordinal Features (ordered categories)

- **Parental_Involvement**: None=0, Low=1, Medium=2, High=3
- **Access_to_Resources**: Low=0, Medium=1, High=2
- **Motivation_Level**: Low=0, Medium=1, High=2
- **Teacher_Quality**: Low=0, Medium=1, High=2
- **Parental_Education_Level**: High School=0, College=1, Postgraduate=2
- **Family_Income**: Low=0, Medium=1, High=2
- **Distance_from_Home**: Near=0, Moderate=1, Far=2

### Categorical Features (one-hot encoded)

- **Gender**: Male, Female
- **School_Type**: Public, Private
- **Extracurricular_Activities**: Yes, No
- **Internet_Access**: Yes, No
- **Peer_Influence**: Positive, Neutral, Negative
- **Learning_Disabilities**: Yes, No

### Numeric Features (standardized)

- **Hours_Studied**: 0-44 hours/week
- **Attendance**: 60-100%
- **Sleep_Hours**: 4-10 hours/night
- **Previous_Scores**: 50-100
- **Tutoring_Sessions**: 0-8 per month
- **Physical_Activity**: 0-6 hours/week

## Benefits

✅ **User-Friendly**: No need to know encoding schemes
✅ **Validated**: Automatic input validation
✅ **Transparent**: Can view raw inputs before preprocessing
✅ **Insightful**: Profile analysis helps interpret results
✅ **Professional**: Clean, organized UI

## Troubleshooting

### Issue: "Error loading preprocessor"

**Solution**: Make sure `Data/Processed/preprocessor.joblib` exists
```bash
# If missing, run preprocessing
python src/data_prep.py
```

### Issue: "Invalid input" error

**Solution**: Check that all required fields are filled and values are within valid ranges

### Issue: Model predictions seem wrong

**Solution**: Verify the preprocessor was trained on the same data as the models
- Check `Data/Processed/` has both preprocessor and train/test splits
- Ensure preprocessing order matches training

## Next Steps

1. ✅ Test the updated dashboard: `streamlit run streamlit_app.py`
2. ✅ Try entering different student profiles
3. ✅ Verify predictions make sense (higher study hours → higher scores)
4. ✅ Use for your presentation demo

---

**The dashboard is now production-ready with a professional user experience!** 🎉

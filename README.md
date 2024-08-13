
---

# NutriPlan AI

NutriPlan AI is a sophisticated meal planning and calorie detection application developed as part of a college project. This innovative tool is designed to help users meet their dietary needs and preferences with ease. Leveraging advanced natural language processing, NutriPlan AI generates creative meal ideas using ingredients selected by the algorithm, ensuring both variety and nutritional balance.

## Features

- **Food Image Detection and Calorie Prediction**: Identify food items and predict their calorie content using a pre-trained TensorFlow model.
- **Daily Calorie Needs Calculation**: Determine daily calorie requirements based on user inputs such as age, height, weight, and gender.
- **Personalized Food Preferences and Restrictions**: Customize meal plans by selecting food preferences and accounting for allergies or dietary restrictions.
- **Meal Plan Generation**: Create balanced meal plans for breakfast, lunch, and dinner within the target calorie ranges.
- **Creative Meal Naming and Descriptions**: Use Anthropic Claude AI to generate creative names and descriptions for meals.
- **User-friendly Interface**: Developed with Streamlit for an intuitive and interactive user experience.

## Technology

- **Python**: Core programming language for the application.
- **Streamlit**: Framework for building the app's UI.
- **Pandas**: Data manipulation library.
- **TensorFlow**: Pre-trained model for food image detection and calorie prediction.
- **Anthropic Claude API**: Natural language processing for generating meal names and descriptions.

## Installation

Clone the repository and install the dependencies using pipenv:

```bash
git clone https://https://github.com/harvinderr15/NutriPlanAI
cd NutriPlanAI
pipenv install
```

## Usage

Run the app using Streamlit:

```bash
streamlit run streamlit_meal_planner.py
```

Add your API keys to `.streamlit/secrets.toml`:

```toml
anthropic_apikey="YOUR_API_KEY"
openai_apikey="YOUR_API_KEY"
```

## Random Greedy Algorithm

The app employs a random greedy algorithm to generate meal plans:

1. Calculate the target calories for breakfast, lunch, and dinner based on the user's Basal Metabolic Rate (BMR).
2. Randomly select a food group (e.g., fruits, proteins).
3. Randomly select a food item from that group.
4. Check if adding the item would exceed the calorie target.
5. If not, add the item to the selected ingredients list.
6. Repeat steps 2-5 until the calories are within 10 of the target or all items are selected.

This approach aims to select a set of food items that maximize calories without exceeding the target, akin to the knapsack problem.

## Project Background

NutriPlan AI was developed as a college project, showcasing the application of artificial intelligence in personalized meal planning and natural language generation. This tool demonstrates significant potential in helping individuals manage their dietary needs efficiently. Future enhancements could include user accounts, expanded food options, and detailed recipe instructions.

## Conclusion

NutriPlan AI is a valuable tool for anyone looking to manage their diet more effectively. Its ability to personalize meal plans and generate creative meal descriptions makes it a practical and enjoyable solution for dietary planning. This project not only highlights the practical applications of AI but also serves as a testament to the skills and innovation fostered during the college project.

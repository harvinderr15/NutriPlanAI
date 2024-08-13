import streamlit as st
import pandas as pd
import random
import time
import openai
from datetime import datetime
import os

import cv2
import numpy as np
from tensorflow.keras.applications import VGG16, ResNet50, MobileNet
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_input_vgg
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_resnet
from tensorflow.keras.applications.mobilenet import preprocess_input as preprocess_input_mobilenet
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import decode_predictions as decode_predictions_vgg
from tensorflow.keras.applications.resnet50 import decode_predictions as decode_predictions_resnet
from tensorflow.keras.applications.mobilenet import decode_predictions as decode_predictions_mobilenet

from data import food_items_breakfast, food_items_lunch, food_items_dinner
from prompts import pre_prompt_b, pre_prompt_l, pre_prompt_d, pre_breakfast, pre_lunch, pre_dinner, end_text, \
    example_response_l, example_response_d, negative_prompt

OPEN_AI_API_KEY = st.secrets["openai_apikey"]
openai.api_key = OPEN_AI_API_KEY
openai.api_base = "https://api.openai.com/v1"

st.set_page_config(page_title="NutriPlanAI", page_icon="🍴")

st.title("NutriPlanAI")

st.divider()

st.write(
    "This is an AI-based meal planner that uses a person's information. The planner can be used to find a meal plan that satisfies the user's calorie and macronutrient requirements.")

st.write("Choose an option:")
option = st.radio("",
                  ("Find Calarioes in your meal", "Find your diet plan"))

if option == "Find Calarioes in your meal":

    uploaded_image = st.file_uploader("Upload an image of a food item")
    if uploaded_image is not None:
        # Load pre-trained models
        model_vgg = VGG16(weights='imagenet', include_top=True)
        model_resnet = ResNet50(weights='imagenet', include_top=True)
        model_mobilenet = MobileNet(weights='imagenet', include_top=True)

        # Dictionary mapping food items to their calorie values (example)
        calories_dict = {
            'pizza': 285,  # Example calorie value for pizza
            'burger': 354,  # Example calorie value for burger
            'salad': 120,
            'hotdog': 150,
            'fries': 222}

        # Function to classify image using a given model
        def classify_image(model, preprocess_input, img_path, decode_predictions):
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            predictions = model.predict(img_array)
            decoded_predictions = decode_predictions(predictions, top=3)[0]  # Get top 1 prediction
            return decoded_predictions

        # Function to display class predictions
        def display_prediction(model_name, predictions):
            print(f"Model: {model_name}")
            for i, (imagenet_id, label, score) in enumerate(predictions):
                print(f"Prediction {i + 1}: {label} (Confidence: {score:.2f})")

        # Function to display class predictions with calorie information
        def display_prediction_with_calories(model_name, predictions):
            result = []
            result.append(f"Model: {model_name}")
            for i, (imagenet_id, label, score) in enumerate(predictions):
                if label.lower() in calories_dict:
                    calorie_value = calories_dict[label.lower()]
                    result.append(
                        f"Prediction {i + 1}: {label} (Confidence: {score:.2f}), Calories: {calorie_value}")
                else:
                    result.append(f"Prediction {i + 1}: {label} (Confidence: {score:.2f}), Calories: Not available")
            return result

        # Path to the image you want to classify
        predicted_classes_resnet = classify_image(model_resnet, preprocess_input_resnet, uploaded_image,
                                                  decode_predictions_resnet)
        predictions_with_calories = display_prediction_with_calories("ResNet50", predicted_classes_resnet)

        # Display predictions with calorie information on Streamlit UI
        for prediction in predictions_with_calories:
            st.write(prediction)


elif option == "Find your diet plan":

    st.write("Enter your information:")
    name = st.text_input("Enter your name")
    dob = st.date_input("Enter your date of birth", min_value=datetime(1950, 1, 1), max_value=datetime.now())
    weight = st.number_input("Enter your weight (kg)")
    height = st.number_input("Enter your height (cm)")
    gender = st.radio("Choose your gender:", ["Male", "Female"])
    example_response = f"This is just an example but use your creativity: You can start with, Hello {name}! I'm thrilled to be your meal planner for the day, and I've crafted a delightful and flavorful meal plan just for you. But fear not, this isn't your ordinary, run-of-the-mill meal plan. It's a culinary adventure designed to keep your taste buds excited while considering the calories you can intake. So, get ready!"

    def calculate_age(dob):
        today = datetime.today()
        age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        return age

    def calculate_bmr(weight, height, dob, gender):
        age = calculate_age(dob)
        if gender == "Male":
            bmr = 9.99 * weight + 6.25 * height - 4.92 * age + 5
        else:
            bmr = 9.99 * weight + 6.25 * height - 4.92 * age - 161

        return bmr

    def get_user_preferences():
        preferences = st.multiselect("Choose your food preferences:", list(food_items_breakfast.keys()))
        return preferences

    def get_user_allergies():
        allergies = st.multiselect("Choose your food allergies:", list(food_items_breakfast.keys()))
        return allergies

    def generate_items_list(target_calories, food_groups):
        calories = 0
        selected_items = []
        total_items = set()
        for foods in food_groups.values():
            total_items.update(foods.keys())

        while abs(calories - target_calories) >= 10 and len(selected_items) < len(total_items):
            group = random.choice(list(food_groups.keys()))
            foods = food_groups[group]
            item = random.choice(list(foods.keys()))

            if item not in selected_items:
                cals = foods[item]
                if calories + cals <= target_calories:
                    selected_items.append(item)
                    calories += cals

        return selected_items, calories

    def knapsack(target_calories, food_groups):
        items = []
        for group, foods in food_groups.items():
            for item, calories in foods.items():
                items.append((calories, item))

        n = len(items)
        dp = [[0 for _ in range(target_calories + 1)] for _ in range(n + 1)]

        for i in range(1, n + 1):
            for j in range(target_calories + 1):
                value, _ = items[i - 1]

                if value > j:
                    dp[i][j] = dp[i - 1][j]
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - value] + value)

        selected_items = []
        j = target_calories
        for i in range(n, 0, -1):
            if dp[i][j] != dp[i - 1][j]:
                _, item = items[i - 1]
                selected_items.append(item)
                j -= items[i - 1][0]

        return selected_items, dp[n][target_calories]

    bmr = calculate_bmr(weight, height, dob, gender)
    round_bmr = round(bmr, 2)
    st.subheader(f"Your daily intake needs to have: {round_bmr} calories")
    choose_algo = "Knapsack"
    if 'clicked' not in st.session_state:
        st.session_state.clicked = False

    def click_button():
        st.session_state.clicked = True

    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-4"

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    hide_streamlit_style = """
                        <style>
                        # MainMenu {visibility: hidden;}
                        footer {visibility: hidden;}
                        footer:after {
                        content:'Software Engineering Project'; 
                        visibility: visible;
                        display: block;
                        position: relative;
                        # background-color: red;
                        padding: 15px;
                        top: 2px;
                        }
                        #ai-meal-planner {
                          text-align: center; !important
                            }
                        </style>
                        """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    st.button("Create a Basket", on_click=click_button)
    if st.session_state.clicked:
        calories_breakfast = round((bmr * 0.5), 2)
        calories_lunch = round((bmr * (1 / 3)), 2)
        calories_dinner = round((bmr * (1 / 6)), 2)

        if choose_algo == "Random Greedy":
            meal_items_morning, cal_m = generate_items_list(calories_breakfast, food_items_breakfast)
            meal_items_lunch, cal_l = generate_items_list(calories_lunch, food_items_lunch)
            meal_items_dinner, cal_d = generate_items_list(calories_dinner, food_items_dinner)

        else:
            meal_items_morning, cal_m = knapsack(int(calories_breakfast), food_items_breakfast)
            meal_items_lunch, cal_l = knapsack(int(calories_lunch), food_items_lunch)
            meal_items_dinner, cal_d = knapsack(int(calories_dinner), food_items_dinner)
        st.header("Your Personalized Meal Plan")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("Calories for Morning: " + str(calories_breakfast))
            st.dataframe(pd.DataFrame({"Morning": meal_items_morning}))
            st.write("Total Calories: " + str(cal_m))

        with col2:
            st.write("Calories for Lunch: " + str(calories_lunch))
            st.dataframe(pd.DataFrame({"Lunch": meal_items_lunch}))
            st.write("Total Calories: " + str(cal_l))

        with col3:
            st.write("Calories for Dinner: " + str(calories_dinner))
            st.dataframe(pd.DataFrame({"Dinner": meal_items_dinner}))
            st.write("Total Calories: " + str(cal_d))
        if st.button("Generate Meal Plan"):
            st.markdown("""---""")
            st.subheader("Breakfast")
            user_content = pre_prompt_b + str(meal_items_morning) + example_response + pre_breakfast + negative_prompt
            temp_messages = [{"role": "user", "content": user_content}]
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                for response in openai.ChatCompletion.create(
                        model=st.session_state["openai_model"],
                        messages=temp_messages,
                        stream=True,
                ):
                    full_response += response.choices[0].delta.get("content", "")
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response})

            st.markdown("""---""")
            st.subheader("Lunch")
            user_content = pre_prompt_l + str(meal_items_lunch) + example_response + pre_lunch + negative_prompt
            temp_messages = [{"role": "user", "content": user_content}]
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                for response in openai.ChatCompletion.create(
                        model=st.session_state["openai_model"],
                        messages=temp_messages,
                        stream=True,
                ):
                    full_response += response.choices[0].delta.get("content", "")
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response})

            st.markdown("""---""")
            st.subheader("Dinner")
            user_content = pre_prompt_d + str(meal_items_dinner) + example_response + pre_dinner + negative_prompt
            temp_messages = [{"role": "user", "content": user_content}]
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                for response in openai.ChatCompletion.create(
                        model=st.session_state["openai_model"],
                        messages=temp_messages,
                        stream=True,
                ):
                    full_response += response.choices[0].delta.get("content", "")
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response})
            st.write("Thank you for using our AI app! I hope you enjoyed it!")


        # Ask user if they followed the plan
        #followed_plan = st.radio("Did you follow the meal plan?", ("","Yes", "No"))
        reward_points=0
        user_data = pd.DataFrame({
                "Name": [name],
                "Date of Birth": [dob],
                "Weight (kg)": [weight],
                "Height (cm)": [height],
                "Gender": [gender],
                "Reward Points": [reward_points]
            })

        st.write("Have you followed the diet plan?")    
        if st.button("Yes", on_click=click_button):
            
        #if followed_plan == "Yes":
            # Generate random reward points
            reward_points = random.randint(1, 100)
            st.write(f"Congratulations! You earned {reward_points} reward points for following the meal plan.")
            user_data['Reward Points'] = reward_points
        elif st.button("NO"):
            
            # Display message for 0 reward points
            st.write("Oops! You got 0 reward points for not following the meal plan.")
            
        user_data_file = 'user_data.xlsx'
        if os.path.isfile(user_data_file):
            existing_data = pd.read_excel(user_data_file)
            if 'Name' in existing_data.columns and name in existing_data['Name'].values:
                existing_reward_points = existing_data[existing_data['Name'] == name]['Reward Points'].values[0]
                user_data['Reward Points'] += existing_reward_points
                existing_data = existing_data[existing_data['Name'] != name]
            combined_data = pd.concat([existing_data, user_data])
            combined_data.to_excel(user_data_file, index=False)
            st.write("Your data has been updated.")
        else:
            user_data.to_excel(user_data_file, index=False)
            st.write("Your data has been saved.")

        # Display updated user information
        st.write("Your information:")
        st.dataframe(user_data)
        
        ranked_data = combined_data.sort_values(by='Reward Points', ascending=False).reset_index(drop=True)
        ranked_data['Date of Birth'] = ranked_data['Date of Birth'].astype(str)
        # Display ranked user data
        st.subheader("Ranked Users by Reward Points")
        st.dataframe(ranked_data)

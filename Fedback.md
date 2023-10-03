### Fail in round(1/3)

# Feedbacks

## App Installation Issues

- **Initial Startup Trouble:** The app failed to start properly the first time I attempted to install its requirements.

- **Include Package Versions:** To ensure consistent behavior, it's advisable to include specific package versions in your requirements file.

## Inconsistent Results

- **Varying Output:** When I ran the app twice, I noticed different plots and answers in each run.

- **First Run Oddity:** The first run produced a smoother prediction line, which did not match the expected plot.

- **Second Run Matching:** However, the second run yielded a plot that closely resembled the expected result.

## Improved Model Handling

- **Model Training:** Consider training a model, saving it, and then loading the saved model for future predictions within your app.

- **Avoid Incomplete Training:** This approach helps avoid scenarios like the ones mentioned earlier, where the training might not have completed correctly.


# Todo
-[x] update requirements.txt with version
-[x] update training main.py so when choose not to update model, it will use the model saved at "model"
-[ ] make the server updating the picture every time it generates a new one

# What I've learned 
## write requirements.txt with versions
In Pycharm, simply delete all the requirements. Using the prompt up option to generate

## Saving, Loading Model when needed
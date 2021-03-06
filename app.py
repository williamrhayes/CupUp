import os
import streamlit as st
from scipy.stats import percentileofscore
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from model.FlavorModel import FlavorModel
import nltk

# Specify the path to get the data
path = os.path.dirname(__file__)

@st.cache
def load_data(path=path):
    nltk.download('stopwords')
    df_all = pd.read_csv(f"{path}/CupOfExcellenceData.txt", delimiter="|")
    df_all.index += 1
    df_rwanda = df_all[df_all['Country'] == 'Rwanda']
    return df_all, df_rwanda

# Generate a scatterplot of the
# coffee scores
def plot_data(df, user_score=None, sample_index=None):

    # Create a copy of the dataframe
    # and specify the data's origin
    df = df.copy()
    df['Coffee From'] = 'Other COE Data'

    # Add the user data to a copy of
    # the composite dataset
    if user_score:
        user_data = {'Farmer': 'You', 'True Score': user_score, 'Coffee From': 'You'}
        df = df.append(user_data, ignore_index=True)

    # Hilight where the sample is
    # in the coffee score distribution
    elif sample_index:
        df.at[sample_index, 'Coffee From'] = f"{df.iloc[sample_index].Country}, {df.iloc[sample_index].Year}"

    df = df.sort_values('True Score', ascending=False).reset_index(drop=True).copy()

    df['All-Time Rank'] = df.index + 1

    # Use the tag so the user can easily spot
    # where their coffee lies among other
    # competing coffees
    if user_score:
        fig = px.bar(df, x="All-Time Rank", y="True Score",
                         color='Coffee From', color_discrete_sequence=["#595959", "#A1FFFF"])

    elif sample_index:
        fig = px.bar(df, x="All-Time Rank", y="True Score",
                         color='Coffee From', color_discrete_sequence=["#595959", "#A1FFFF"])

    else:
        fig = px.bar(df, x="All-Time Rank", y="True Score", color='True Score')

    fig.update_traces(marker_line_width=0)

    # Add a plot title
    fig.update_layout(
        yaxis_range=[84, 95],
        title={
            'text': "Coffee Score Distribution",
            'y': 0.925,
            'x': 0.475,
            'xanchor': 'center',
            'yanchor': 'top'})

    return fig

# Generate a bar chart displaying the difference between
# the model prediction and the true score
def compare_scores_bar_chart(df):
    fig = go.Figure(data=[
        go.Bar(name='Actual', x=['Score'], y=df['True Score']),
        go.Bar(name='Predicted', x=['Score'], y=df['Predicted Score'])
    ])

    fig.update_layout(barmode='group',
                      yaxis_range=[84, 96],
                      yaxis_title="Cupping Score",
                      title={'text': "True Score vs Model Prediction",
                             'y':0.9,
                             'x':0.475,
                             'xanchor': 'center',
                             'yanchor': 'top'})

    return fig

# Generate a bar chart displaying the difference between
# the baseline error and the XGBoost model error
# Found in the ("Results" section)
def compare_errors_bar_chart():
    fig = go.Figure(data=[
        go.Bar(name='Baseline', x=['Baseline Error'], y=[1.48]),
        go.Bar(name='ML Model', x=['Model Error'], y=[1.16])
    ])

    fig.update_layout(barmode='group',
                      yaxis_range=[0, 2],
                      yaxis_title="MAE (Cupping Points)",
                      title={'text': "Mean Absolute Error (MAE) Comparison",
                             'y':0.9,
                             'x':0.475,
                             'xanchor': 'center',
                             'yanchor': 'top'})

    return fig

# Read in the data into two distinct
# dataframes (composite and Rwandan)
df, df_rwanda = load_data()

# Get the files necessary to predict the score
count_vectorizer = f"{path}/model/CountVectorizer.cv"
xgb_model = f"{path}/model/Model.xgb"

# Construct the model object
model = FlavorModel(count_vectorizer, xgb_model)

# Set the title
st.title(':coffee: Rwanda Deluxe Insights: CupUp :coffee:')

# Build the sidebar
dataset = st.sidebar.selectbox('Pull a random entry from...',
                          ('All Coffees', 'Rwandan Coffees'))
show_sample = st.sidebar.button('Show Me a Sample')
st.sidebar.write('Want to Learn More?')
display_stats = st.sidebar.button('Yes! How Does This Thing Work?')
st.sidebar.write('Need Coffee?')
st.sidebar.markdown('[Shop Rwanda Deluxe](https://www.rwandadeluxecoffee.com/shop)', unsafe_allow_html=True)


# Build the coffee description input
flavor_presets = st.multiselect('Your coffee has notes of', ["almond","apple","apricot","banana","bergamot","berries","berry","blackberry","blackcherry","blackcurrant","blacktea","blueberry","brownsugar","butter","butterscotch","buttery","caramel","cedar","cherries","cherry","chocolate","cinnamon","citric","citrus","clove","cocoa","coconut","cranberry","darkchocolate","driedfruit","fig","floral","fruit","ginger","grape","grapefruit","grapes","greenapple","greentea","guava","hazelnut","herbal","hibiscus","honey","jasmine","juicy","kiwi","lavender","lemon","lemongrass","lime","lychee","malic","malt","mandarin","mango","maplesyrup","marzipan","melon","milkchocolate","mint","molasses","nectarine","nuts","nutty","orange","orangepeel","papaya","passionfruit","peach","pear","perfume","pineapple","plum","prune","raisin","raisins","raspberry","redapple","redcurrant","redgrape","redwine","rose","spice","spicy","stonefruit","strawberry","sugarcane","sweet","tamarind","tangerine","tobacco","toffee","tropicalfruit","vanilla","walnut","watermelon","wine"])
coffee_desc = st.text_input("Describe any additional flavors, aromas, and textures of your coffee...")

# Place the "Judge Coffee" button in the center
_, center1, _ = st.beta_columns((1, 0.7, 1))
with center1:
    judge_coffee = st.button("Send to Judges")

# SEND TO JUDGES BUTTON
if ((judge_coffee or coffee_desc) and not
    (show_sample or display_stats)):

    # Add pre-selected flavors to the guess
    if flavor_presets:
        coffee_desc = ', '.join(flavor_presets) + ', ' + coffee_desc

    # If the user entered no input
    if coffee_desc == '':
        _, center2, _ = st.beta_columns((1, 0.5, 1))
        with center2:
            st.write('Score: 85.50')
        st.write('No flavor input, displaying the baseline score')

    # If the user entered the right answer
    elif coffee_desc.lower() == 'rwanda deluxe coffee':
        _, center2, _ = st.beta_columns((1, 0.5, 1))
        with center2:
            st.write('Good answer!')
        results = pd.DataFrame(data={'Predicted Score': ['ERROR: Integer Overload'],
                                     'Competitors Defeated': ['ERROR: Integer Overload'],
                                     'Percentile': [101],
                                     'Eternal Title': ['Ultimate Coffee God for All Eternity']})
        results_chart = st.write(results)

    # SEND TO JUDGES BUTTON IS PRESSED
    elif judge_coffee:

        # Predict the score using our model
        user_score = round(model.predict(coffee_desc),2)

        # Calculate the score change from
        # the model's baseline prediction of 85.5
        change = str(round(user_score - 85.5, 2))

        # Calculate the score's percentile,
        # rounded to two significant figures
        percentile = str(int(round(percentileofscore(df['True Score'].values, user_score), 0)))

        # Calculate the number of cups this flavor
        # profile defeated
        defeated_cups = int((1 - user_score / 100) * len(df))

        # Calculate the all-time rank based on this score
        place = int((user_score / 100) * len(df))

        # Consolidate results into a dataframe to display to
        # the user
        results = pd.DataFrame(
            data={'Predicted Score': [user_score],
                  'Baseline Improvement': [change],
                  'Competitors Defeated': [defeated_cups],
                  'Percentile': [percentile]
                  })

        # Find the place
        results.index += place

        # Place the user's score in the center of the
        # page as an output
        _, center2, _ = st.beta_columns((1, 0.5, 1))
        with center2:
            st.write(f'Score: {str(round(user_score, 2))}')

        # Remind the user of their input
        st.write(f'Description: {coffee_desc}')

        # Display the table of results
        st.write(results)

        # Plot the user's result
        # along with all the other coffees
        fig = plot_data(df, user_score=user_score)
        st.plotly_chart(fig)



# SHOW SAMPLE BUTTON
if (show_sample and not
    (judge_coffee or display_stats)):

    # If the "All Coffees" coffee preset is selected
    if dataset == 'All Coffees':
        # Choose a random coffee from the dataset by index
        choice_index, = np.random.choice(df.index.values, 1)
        sample_index = choice_index

    # If the "Rwandan Coffees" preset is selceted in the sidebar
    elif dataset == 'Rwandan Coffees':
        # Choose a random Rwandan coffee from the dataset by index
        choice_index, = np.random.choice(df[df['Country'] == 'Rwanda'].index.values, 1)
        sample_index = choice_index - 1
        choice_index -= 1

    choice_values = df.iloc[choice_index]
    true_score = choice_values['True Score']
    predicted_score = round(choice_values['Predicted Score'], 2)
    percentile = percentileofscore(df['True Score'].values, true_score)

    # Calculate the error between what our model predicted the
    # score would be and what the true score was
    error = round(choice_values['Model Error'], 2)

    # Display the True Score to the user
    _, center2, _ = st.beta_columns((1, 0.5, 1))
    with center2:
        st.write(f'Score: {str(round(true_score, 2))}')

    st.write(f'Description: {choice_values.Characteristics}')

    choice_df = df.iloc[choice_index:choice_index+1]
    chart = st.write(choice_df
                     [['Country', 'Year', 'Farmer',
                       'Predicted Score', 'True Score',
                       'Model Error']])

    # Plot the sample results
    # along with all the other coffees
    fig = plot_data(df, sample_index=sample_index)
    st.plotly_chart(fig)

    bar_fig = compare_scores_bar_chart(choice_df)

    st.plotly_chart(bar_fig)

    # Show the country of origin on a map
    if dataset == 'All Coffees':
        st.title(f"Country of Origin: {df.iloc[choice_index].Country}")
        st.map(df.iloc[choice_index:choice_index+1], zoom=1)

    ### WORK IN PROGRESS ###
    # Show the specific origin region in Rwanda
    elif dataset == 'Rwandan Coffees':
        st.title(f"Country of Origin: {df.iloc[choice_index].Country}")
        st.map(df.iloc[choice_index:choice_index+1], zoom=6.5)

# TELL ME ABOUT THE DATA BUTTON
if display_stats:

    # Full write-up

    st.title('Where Does My Score Come From?')
    st.markdown("When you enter a flavor description into the input bar above, it isn???t just spitting out a random number. The score you see is the output of a machine learning model trained on 15 years of competitive coffee-cupping data collected from the [Cup of Excellence website] (https://cupofexcellence.org/).")

    st.title('The Cup of Excellence')
    st.markdown("The Cup of Excellence is a charity that promotes the education of coffee farmers in countries throughout Africa and the Americas. ")
    st.map(df.drop_duplicates('Country'))
    st.markdown("Each year the Cup of Excellence hosts a competition where coffees are rigorously tested and scored in an [intense vetting process](https://cupofexcellence.org/rules-protocols/). In order to be considered a ???Cup of Excellence??? a cup of coffee must score higher than 86 out of 100 points in four consecutive rounds among a variety of judges. The top 10 cups from these competitions are scored one last time to finalize their score and ranking. After coffees are scored the green beans are put up for auction.")
    st.markdown("These competitions have been running since 1999 and in addition to enriching the lives of coffee farmers, have also produced a [good chunk of data](https://cupofexcellence.org/farm-directory/) about what makes a phenomenal cup of coffee. To improve our understanding of what makes world-class coffee, we consolidated this data and asked a simple question...")

    st.title("Can a Coffee's Flavor Profile Be Used to Accurately Predict Its Cupping Score?")
    st.markdown("Roughly 3,000 of the 4,200 competition results included flavor descriptions for their coffee. We can see these coffees sorted by score and graphed below.")
    fig = plot_data(df)
    st.plotly_chart(fig)
    st.markdown("The flavors describing these coffees are adjectives such as 'chocolate', 'caramel', and 'vanilla' as well as more nuanced descriptors such as 'elegant' and ['complex'](https://www.rwandadeluxecoffee.com/rwanda-deluxe-insights). We wanted to build a model that could take these flavor descriptions as input and output the score of the coffee. First, we needed a starting point to compare to.")

    st.title("Establishing a Baseline")
    st.markdown("In order to track our progress (and make sure the flavors actually improve our prediction accuracy) we needed to choose a baseline model. A baseline model is the simplest prediction method we can reasonably make. For this analysis our baseline model was to predict that any coffee, regardless of its flavor description, would get the average score every time. Not a very exciting model, but that's what makes it a great starting point. When we guess the average for each cup of coffee regardless of its flavor description we have an **average error of 1.48 points**.")

    st.title("Building Our Model")
    st.markdown("We built (and deployed) our model using Python. Specifically we used a word-stemmer to represent multiple words with the same meaning (such as peach, peaches, and peachy) as one word (peachi). We then used a count-vectorizer to turn the words into a vector (where each word could be represented as a number) and XGBoost to predict the score of the coffee from this vector. A few other deep-learning techniques were attempted however none could out-perform XGBoost. Once we decided to proceed with XGBoost, we attempted hundreds of different parameter combinations to optimize for the lowest mean absolute error.")

    st.title("Evaluating Our Model (Results)")
    st.markdown('We achieved a Mean Absolute Error of 1.16 points on our test dataset using the XGBoost model. This means that on average, this prediction is off by about 1.16 points. Based on subjective flavor profiles alone **we were able to improve our prediction accuracy from the baseline by about 22%**.')
    fig = compare_errors_bar_chart()
    st.plotly_chart(fig)

    st.title("Conclusion")
    st.markdown("There certainly seems to be a method to the madness found in coffee flavor descriptions. We were able to make more educated guesses about the quality of the coffee based on its description alone. Of course, there are some important notes to keep in mind here:")
    st.markdown("* First, the scores may fluctuate over time. Some years may just be judged more harshly than others and it is difficult to say whether a cup that scored 94 points in 1999 would score 94 points today.")
    st.markdown("* Second, the scores from this dataset are skewed much higher than the normal every-day cup of coffee. After all, at the end of the day each coffee in this dataset is considered a ???Cup of Excellence???. This means that the output is already assuming you are already a top-notch coffee connoisseur.")
    st.markdown("* Third, the Cup of Excellence is not the be-all end-all judgement of excellent coffee. There may be flavors, aromas, and textures that the judges don???t like that many people do like. Coffee tasting will always be a subjective experience and this model is better at predicting what these judges might enjoy.")
    st.markdown("There are many more objective factors that could dramatically improve our prediction accuracy. Factors such as, growing altitude, country of origin, and coffee variety (I???m looking at you Geisha!) could offer significant insight into how to accurately predict the score of a given coffee. Of course our most promising results are kept an internal secret so that Rwanda Deluxe Coffee can gain a competitive edge. We appreciate your interest and hope you enjoyed reading about our research.")


# Rigirous testing rules for COE - https://cupofexcellence.org/rules-protocols/
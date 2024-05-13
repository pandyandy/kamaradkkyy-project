import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.express as px
import openai
from openai import OpenAI
from wordcloud import WordCloud
from kbcstorage.client import Client

openai.api_key = st.secrets["OPENAI_API_KEY"]

client = Client(st.secrets['keboola_url'], st.secrets['keboola_token'])

def generate(content):
    client = OpenAI()

    completion = client.chat.completions.create(
    model="gpt-4-turbo-preview",
    messages=[
        {"role": "system", "content": "You are given a review for Kistamässan Exhibition & Congress Centre in Sweden. Pretend you are the owner of the conference centre and reply to it in English."},
        {"role": "user", "content": content}
    ]
)
    return completion.choices[0].message.content

if 'response' not in st.session_state:
  	st.session_state['response'] = ""
    
# Load the data
data = pd.read_csv('/data/in/tables/consolidated_analysis_reviews.csv')
keywords = pd.read_csv('/data/in/tables/grouped_keywords_reviews.csv')

# Add widget column for st.data_editor
data['is_widget'] = False


# Logo and Title
keboola_logo = '/data/in/files/518615_logo.png'
keboola_logo_html = f'<div style="display: flex; justify-content: flex-end;"><img src="data:image/png;base64,{base64.b64encode(open(keboola_logo, "rb").read()).decode()}" style="width: 150px; margin-left: -10px;"></div>'
st.markdown(f'{keboola_logo_html}', unsafe_allow_html=True)

st.title('Review Explorer')

# Filter
min_score, max_score = st.slider(
            'Select a range for the sentiment score:',
            min_value=-1.0, max_value=1.0, value=(-1.0, 1.0)
        )

filtered_data = data[(data['sentiment_score'] >= min_score) & (data['sentiment_score'] <= max_score)]
keywords_filtered = keywords[(keywords['sentiment_score'] >= min_score) & (keywords['sentiment_score'] <= max_score)]

# Show table
selected_data = st.data_editor(filtered_data[['is_widget',
  													'name',
                            'rating', 
                            'sentiment_score',
                            'text', 
                            'review_link']], 
             column_config={'is_widget': 'Select',
               							'name': 'Name', 
                            'rating': 'Rating',
                            'sentiment_score': 'Sentiment Score',
                            'text': 'Text',
                            'review_link': st.column_config.LinkColumn('URL')
                            },
            disabled=['name',
                    	'rating', 
                      'text', 
                      'sentiment_score', 
                      'review_link'],
            use_container_width=True, hide_index=True)
    
# Histogram
fig = px.histogram(filtered_data, x='sentiment_score', nbins=20, title='Distribution of Sentiment Score')

fig.update_layout(
    xaxis_title='Sentiment Score',
    yaxis_title='Count',
    bargap=0.1
)
st.plotly_chart(fig, use_container_width=True)

# Wordcloud
word_freq = keywords_filtered.set_index('keyword')['keyword_count'].to_dict()
colormap = mcolors.ListedColormap(['#0069c8', '#85c9fe', '#ff2a2b', '#feaaab', '#2bb19d'])

title_text = 'Keyword Frequency'
st.markdown(f'<br>**{title_text}**', unsafe_allow_html=True)

wordcloud = WordCloud(width=800, height=400, background_color=None, mode='RGBA', colormap=colormap).generate_from_frequencies(word_freq)
wordcloud_array = wordcloud.to_array()

plt.figure(figsize=(10, 5), frameon=False)
plt.imshow(wordcloud_array, interpolation='bilinear')
plt.axis('off')
st.pyplot(plt)

# Gemini response
keboola_openai = '/data/in/files/526193_keboola_openai.png'
openai_html = f'<div style="display: flex; justify-content: flex-end;"><img src="data:image/png;base64,{base64.b64encode(open(keboola_openai, "rb").read()).decode()}" style="width: 60px; margin-top: 30px;"></div>'
st.markdown(f'{openai_html}', unsafe_allow_html=True)

st.markdown("""
<div style="text-align: left;">
    <h4>Reply to a review with AI</h4>
</div>
""", unsafe_allow_html=True)

if selected_data['is_widget'].sum() == 1:
    data_ai = selected_data[selected_data['is_widget'] == True]['text']
    review_text = data_ai.iloc[0] if not data_ai.empty else st.warning('No review found.')
    st.write(f'_Review:_\n\n{review_text}')

    selected_row = selected_data[selected_data['is_widget'] == True]
    review_key = selected_row['name'].iloc[0]

    if st.button('Generate response'):
        with st.spinner('🤖 Generating response, please wait...'):
            prompt = f"""
            Write a short (2-3 sentences) reply in English.

            Review:
            {review_text}
            """
            st.session_state[review_key] = generate(prompt)

    if review_key in st.session_state:
        st.write(f"_Response:_\n\n{st.session_state[review_key]}")

        # Save to Keboola
        if st.button('Upload to Keboola'):
            with st.spinner("📤 Uploading data, please wait..."):
                identifier = review_key

                matching_row = data[data['name'] == identifier].copy()
                matching_row.loc[:, 'gemini_response'] = st.session_state.get(review_key, '')
                matching_row = matching_row.drop(columns=['is_widget'])

                matching_row.to_csv('review_responses.csv', index=False)
                try:
                    client.tables.load(table_id='out.c-reviews.review_responses', file_path='review_responses.csv', is_incremental=False)
                    st.success('Uploaded.')
                except Exception as e:
                    st.error(f"Data upload failed with: {str(e)}")
else:
    st.info('Select the review you want to respond to in the table above.')

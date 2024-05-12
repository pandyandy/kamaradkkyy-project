import streamlit as st
import pandas as pd
import base64
import jwt
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.express as px

from google.oauth2 import service_account
from wordcloud import WordCloud
from vertexai.generative_models import GenerativeModel
from kbcstorage.client import Client

PROJECT = 'keboola-ai'
LOCATION = 'us-central1'
MODEL_NAME = 'gemini-1.5-pro-preview-0409'

CREDENTIALS = service_account.Credentials.from_service_account_info(
    jwt.decode(st.secrets['encoded_token'], 'keboola', algorithms=['HS256'])
)

client = Client(st.secrets['keboola_url'], st.secrets['keboola_token'])

# Gemini 
def generate(content):
    vertexai.init(project=PROJECT, location=LOCATION, credentials=CREDENTIALS)
    model = GenerativeModel(MODEL_NAME)

    config = {
        'max_output_tokens': 8192,
        'temperature': 1,
        'top_p': 0.95,
    }
    
    responses = model.generate_content(
        contents=content,
        generation_config=config,
        stream=True,
    )
    return "".join(response.text for response in responses)

if 'response' not in st.session_state:
  	st.session_state['response'] = ""
    
# Logos
keboola_logo = '/data/in/files/296014_logo.png'
#qr_code = '/data/in/files/1112988135_qr_code.png'
keboola_gemini = '/data/in/files/296015_keboola_gemini.png'

# Load the data
data = pd.read_csv('/data/in/tables/consolidated_analysis_reviews.csv')
keywords = pd.read_csv('/data/in/tables/grouped_keywords_reviews.csv')

# Add widget column for st.data_editor
data['is_widget'] = False

# Sidebar
#st.sidebar.markdown("""
#<div style="text-align: center;">
#    <h1>GCP Data Cloud Live</h1>
#    <br><p>Scan the QR code to see yourself on the dashboard:</p>
#</div>
#""", unsafe_allow_html=True)
#qr_html = f'<div style="display: flex; justify-content: center;"><img src="data:image/png;base64,{base64.b64encode(open(qr_code, "rb").read()).decode()}" style="width: 200px;"></div>'
#st.sidebar.markdown(f'{qr_html}', unsafe_allow_html=True)
#st.sidebar.markdown('<div style="text-align: center"><br><br><br>Get in touch with Keboola: <a href="https://bit.ly/cxo-summit-2024">https://bit.ly/cxo-summit-2024</a>#replace</div>', unsafe_allow_html=True)

# Title and Filters
keboola_logo_html = f'<div style="display: flex; justify-content: flex-end;"><img src="data:image/png;base64,{base64.b64encode(open(keboola_logo, "rb").read()).decode()}" style="width: 150px; margin-left: -10px;"></div>'
st.markdown(f'{keboola_logo_html}', unsafe_allow_html=True)

st.title('Review Explorer')

min_score, max_score = st.slider(
            'Select a range for the sentiment score:',
            min_value=-1.0, max_value=1.0, value=(-1.0, 1.0)
        )

# Apply Filters
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
gemini_html = f'<div style="display: flex; justify-content: flex-end;"><img src="data:image/png;base64,{base64.b64encode(open(keboola_gemini, "rb").read()).decode()}" style="width: 60px; margin-top: 30px;"></div>'
st.markdown(f'{gemini_html}', unsafe_allow_html=True)

st.markdown("""
<div style="text-align: left;">
    <h4>Reply to a review with Gemini</h4>
</div>
""", unsafe_allow_html=True)

if selected_data['is_widget'].sum() == 1:
    gemini_data = selected_data[selected_data['is_widget'] == True]['text']
    review_text = gemini_data.iloc[0] if not gemini_data.empty else st.warning('No review found.')
    st.write(f'_Review:_\n\n{review_text}')

    selected_row = selected_data[selected_data['is_widget'] == True]
    review_key = selected_row['name'].iloc[0]

    if st.button('Generate response'):
        with st.spinner('🤖 Generating response, please wait...'):
            prompt = f"""
            You are given a review for Kistamässan Exhibition & Congress Centre in Sweden. Pretend you are the owner of the conference centre and write a short reply in English.

            Review:
            {review_text}
            """
            st.session_state[review_key] = generate(prompt)  # Store the response with the review as the key

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

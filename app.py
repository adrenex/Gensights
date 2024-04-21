from flask import Flask, request, render_template, Response
from flask_mail import Mail, Message
import requests, time, os, json, re
import pandas as pd
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_anonymizer import AnonymizerEngine
import warnings
warnings.filterwarnings('ignore')
from openai import OpenAI
import boto3
from dotenv import load_dotenv
load_dotenv()
from google_play_scraper import Sort, reviews
from pymongo import MongoClient

# Configuring Flask
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "secret key"

app.config['MAIL_SERVER']='smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = os.getenv("SENDER_MAIL")
app.config['MAIL_PASSWORD'] = os.getenv("SENDER_PASSWORD")
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True
mail = Mail(app)

def scrape_reviews(url):
    app_reviews = reviews(
        url,
        lang='en', # defaults to 'en'
        country='in', # defaults to 'us'
        count=10,
        sort=Sort.NEWEST, # defaults to Sort.MOST_RELEVANT
    ) 

    review_df = pd.DataFrame(app_reviews[0])
    print("Number of Reviews: ", len(review_df))
    print(review_df.head() )
    return review_df

def inject_reviews_in_mongo(app, review_df):

    mongo_uri = "mongodb+srv://nileshpopli:easysapassword@cluster0.rotzjrk.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    client = MongoClient(mongo_uri)
    db = client[app]
    collection = db['reviews']  # Replace 'your_collection_name' with your actual collection name

    existing_data = collection.find()
    existing__data_df = pd.DataFrame(list(existing_data))
    existing_review_ids = set()

    if existing__data_df.empty:
        print("Previously no data")

    if not existing__data_df.empty:
        existing_review_ids = set(existing__data_df['reviewId'])
        latest_timestamp = collection.find_one(sort=[("at", -1)])['at']
        print("Last injested review:", latest_timestamp)

    # Initialize an empty list to store rows to be removed
    rows_to_remove = []

    for index, row in review_df.iterrows():
        if row['reviewId'] in existing_review_ids:
            rows_to_remove.append(index)

    # Remove rows from new_df based on the list of indices to be removed
    review_df.drop(rows_to_remove, inplace=True)
    
    print("New Rows")
    print(review_df)

    # Convert new rows to dictionary records
    new_data = review_df.to_dict(orient='records')

    # Print the number of new rows
    num_new_rows = len(new_data)
    print(f"Number of new rows to be inserted: {num_new_rows}")

    # Insert only the new rows into MongoDB
    if num_new_rows > 0:
        collection.insert_many(new_data)
        print("New reviews inserted successfully into MongoDB.")
    else:
        print("No new data to insert.")

    all_data = collection.find()
    all_data_df = pd.DataFrame(list(all_data))

    client.close()
    return all_data_df, review_df

def review_etl_pipeline(app_url, app_name):
    review_df = scrape_reviews(app_url)
    #transform()
    temp_df = review_df.copy()
    temp_df['repliedAt'].fillna('Unknown', inplace=True)
    all_data_df, new_reviews_df = inject_reviews_in_mongo(app_name, temp_df)
    return all_data_df, new_reviews_df

def initialize_api_client():
    return OpenAI(
        api_key = os.getenv('OPENAI_API_KEY')
    )

def get_completion_from_messages(system_prompt, user_prompt, review, model="gpt-3.5-turbo", temperature=0.4):

    client = initialize_api_client()
    chat_completion = client.chat.completions.create(
        messages =
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt + f"Review: {review}"},
        ],
        model=model,
        temperature=temperature
        
    )

    return chat_completion.choices[0].message.content

def get_prompt():
    system_prompt = """Analyze the provided Google Play Store review and extract the following insights:
    1) Sentiment: Determine the overall sentiment of the review, classifying as "Positive", "Negative", or "Neutral".
    2) Rating: Rate the review provided by the user, ranging from 1 to 5 stars.
    3) Issue Existence: Identify if there is any issue mentioned in the review. Classify as "Yes" or "No".
    4) Issue Type: If an issue is identified, categorize it into types such as "App praise", "Product praise", "Service praise", etc.
    5) Issue Details: Provide additional details regarding the identified issue, if available.
    6) Paise Existence: Identify if there is any Praise mentioned in the review. Classify as "Yes" or "No".
    7) Praise Type: If an Praise is identified, categorize it into types such as "App praise", "Product praise", "Service praise", etc.
    8) Praise Details: Provide additional details regarding the identified Praise, if available.
    9) Feature Request: Determine if the review includes any feature requests. Classify as "Yes" or "No".
    10) User Loyalty: Assess the user's likelihood to continue using the app based on their review. Classify as "Likely", "Unlikely", or "Neutral".
    11) User Profile: Gather demographic information about the user if available, such as age, gender, location, etc.
    12) User Activity Level: Determine the user's activity level within the app based on their review. Classify as "High", "Medium", or "Low".
    13) User Experience: Evaluate the overall user experience mentioned in the review.
    14) User Suggestions: Identify any suggestions provided by the user for improvement.
    15) User Satisfaction: Gauge the overall satisfaction of the user based on their review. Classify as "Satisfied", "Neutral", or "Not Satisfied".
    16) Promoter Intent: Determine whether the user is likely to recommend the app to others. Classify as "Promoter", "Passive", or "Detractor".
    17) User Status: Identify if the user is a new user, old user, or somewhere in between.
    Output format:
    {
        "sentiment": "Positive / Negative / Neutral",
        "rating": "1-5",
        "issue_existence": "Yes / No",
        "issue": [
            {
                "issue_type": "string",
                "issue_details": "string"
            }
        ],
        "praise_existence": "Yes / No",
        "praise": [
            {
                "praise_type": "string",
                "praise_details": "string"
            }
        ],
        "update_mention": true // or false,
        "feature_request": true // or false,
        "user_loyalty": "Likely / Unlikely / Neutral",
        "user_profile": {
            "age": "string", // If not mentioned, then "Not Mentioned"
            "gender": "string", // If not mentioned, then "Not Mentioned"
            "location": "string" // If not mentioned, then "Not Mentioned"
        },
        "user_activity_level": "High / Medium / Low",
        "user_experience": "string",
        "user_suggestions": "string",
        "user_satisfaction": "Satisfied / Neutral / Not Satisfied",
        "promoter_intent": "Promoter / Passive / Detractor",
        "user_status": "New User / Old User / In Between"
    }
    """

    user_prompt = """Please respond in English: 'en'."""

    return system_prompt, user_prompt

def extract_review_insights(new_reviews_df):
    list_of_reviews = new_reviews_df['content'].tolist()
    gpt_responses = []
    system_prompt, user_prompt = get_prompt()

    i=0
    for review in list_of_reviews:
        gpt_response = get_completion_from_messages(system_prompt, user_prompt, review)
        gpt_responses.append(gpt_response)
        i+=1
        print(i)
    return gpt_responses, list_of_reviews

def parse_google_play_review(response):
    response_json = json.loads(response)
    
    sentiment = response_json['sentiment']
    rating = response_json['rating']
    issue_existence = response_json['issue_existence']
    
    # Extracting issue details
    issue_list = []
    for issue_entry in response_json.get('issue', []):
        issue_list.append({
            "issue_type": issue_entry.get('issue_type', ''),
            "issue_details": issue_entry.get('issue_details', '')
        })
    
    praise_existence = response_json['praise_existence']
    
    # Extracting issue details
    praise_list = []
    for praise_entry in response_json.get('praise', []):
        praise_list.append({
            "praise_type": praise_entry.get('praise_type', ''),
            "praise_details": praise_entry.get('praise_details', '')
        })

    update_mention = response_json['update_mention']
    feature_request = response_json['feature_request']
    user_loyalty = response_json['user_loyalty']
    
    # Extracting user profile
    user_profile = {
        "age": response_json['user_profile'].get('age', 'Not Mentioned'),
        "gender": response_json['user_profile'].get('gender', 'Not Mentioned'),
        "location": response_json['user_profile'].get('location', 'Not Mentioned')
    }

    user_activity_level = response_json.get('user_activity_level', '')
    user_experience = response_json.get('user_experience', '')
    user_suggestions = response_json.get('user_suggestions', '')
    user_satisfaction = response_json.get('user_satisfaction', '')
    promoter_intent = response_json.get('promoter_intent', '')
    user_status = response_json.get('user_status', '')

    return (
        sentiment, rating, issue_existence, issue_list,
        praise_existence, praise_list,
        update_mention, feature_request, user_loyalty, 
        user_profile, user_activity_level, user_experience, 
        user_suggestions, user_satisfaction, promoter_intent, user_status
    )

def transform_insight_for_one(temp_df, review, gpt_response):

    (sentiment, rating, issue_existence, issue_list, praise_existence, praise_list, update_mention, feature_request, user_loyalty, 
    user_profile, user_activity_level, user_experience, user_suggestions, 
    user_satisfaction, promoter_intent, user_status) = parse_google_play_review(gpt_response)

    new_row = {
        'Review': [review],
        'Sentiment': [sentiment],
        'Rating': [rating],
        'Issue_Existence': [issue_existence],
        'Issue(s)': [issue_list],
        'Praise_Existence': [praise_existence],
        'Praise(s)': [praise_list],
        'Update_Mention': [update_mention],
        'Feature_Request': [feature_request],
        'User_Loyalty': [user_loyalty],
        'User_Profile': [user_profile],
        'User_Activity_Level': [user_activity_level],
        'User_Experience': [user_experience],
        'User_Suggestions': [user_suggestions],
        'User_Satisfaction': [user_satisfaction],
        'Promoter_Intent': [promoter_intent],
        'User_Status': [user_status]
    }
    # print(temp_df)
    # print(type(temp_df))
    # print(new_row)

    # Append the new row to the DataFrame
    new_df = pd.DataFrame(new_row)
    result_df = pd.concat([temp_df, new_df], ignore_index=True)
    return result_df

def transform_review_insights_for_all(gpt_responses, list_of_reviews):
    review_insights_df = pd.DataFrame(columns=['Review', 'Sentiment', 'Rating', 'Issue_Existence', 'Issue(s)', 'Praise_Existence',
                                'Praise(s)', 'Update_Mention', 'Feature_Request', 'User_Loyalty', 'User_Profile', 'User_Activity_Level', 
                                'User_Experience', 'User_Suggestions', 'User_Satisfaction', 'Promoter_Intent', 'User_Status'])

    for gpt_response, review in zip(gpt_responses, list_of_reviews):
        review_insights_df = transform_insight_for_one(review_insights_df, review, gpt_response)
    print(review_insights_df)
    return review_insights_df

def inject_review_insights_in_mongo(app, review_df):

    mongo_uri = "mongodb+srv://nileshpopli:easysapassword@cluster0.rotzjrk.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    client = MongoClient(mongo_uri)
    db = client[app]
    collection = db['review_insights']  # Replace 'your_collection_name' with your actual collection name

    # Convert new rows to dictionary records
    data = review_df.to_dict(orient='records')

    # Print the number of new rows
    num_new_rows = len(data)
    print(f"Number of new rows to be inserted: {num_new_rows}")

    # Insert only the new rows into MongoDB
    if num_new_rows > 0:
        collection.insert_many(data)
        print("New insights inserted successfully into MongoDB")
    else:
        print("No new data to insert.")

    all_data = collection.find()
    all_data_df = pd.DataFrame(list(all_data))

    client.close()
    return all_data_df

def review_insights_pipeline(app, new_reviews_df):
    gpt_responses, list_of_reviews = extract_review_insights(new_reviews_df)
    review_insights_df = transform_review_insights_for_all(gpt_responses, list_of_reviews)
    all_review_insights_df = inject_review_insights_in_mongo(app, review_insights_df)
    review_insights_df.to_csv("Review_Insights.csv", index=False)
    return review_insights_df, all_review_insights_df

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def extract_audios(folder_prefix):

    # Explicitly pass AWS credentials
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')

    # Create an S3 client
    s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, region_name='us-east-1')

    # Specify the bucket name and the folder prefix
    bucket_name = 'interaction-audios'

    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder_prefix)

    file_data_list = []
    if 'Contents' in response:
        for obj in response['Contents']:
            # Fetch each object in the folder
            object_key = obj['Key']
            response = s3.get_object(Bucket=bucket_name, Key=object_key)
            
            # Read the object's data as bytes
            byte_data = response['Body'].read()
            
            # Add file name and byte data to the list
            file_data_list.append((object_key.split('/')[-1], byte_data))
    else:
        print(f"No objects found in the specified folder: {folder_prefix}")
    
    file_data_list.pop(0)

    # Print the list of file names and byte data
    for file_name, byte_data in file_data_list:
        print(f"File Name: {file_name}")

    return file_data_list

def iniialize_payload(language = 'en'):

    JWT = os.getenv('JWT')
    
    # voicegain
    host = "https://api.voicegain.ai/v1"
    data_url = "{}/data/file".format(host)

    #host="voicegain"
    headers = {"Authorization": JWT}

    audio_type = "audio/wav"

    asr_body = {
        "sessions": [{
        "asyncMode": "OFF-LINE",
        "poll": {
            "persist": 120000
        },
        "content": {
            "incremental": ["progress"],
            "full" : ["words","transcript"]
        }
        }],
        "audio":{
            "source": {
                "dataStore": {
                    "uuid": "<data-object-UUID>"
                }
            }
        },
        "settings": {
            "asr": {
                "acousticModelNonRealTime" : "whisper",
                "languages" : [language],
            }
        }
    }
    return data_url, headers, audio_type, asr_body, host

def helper(asr_body,headers,fname,host):
    
    #print("making asr request for file {fname}", flush=True)
    #print(asr_body)
    asr_response = requests.post("{}/asr/transcribe/async".format(host), json=asr_body, headers=headers).json()
    print(asr_response)
    session_id = asr_response["sessions"][0]["sessionId"]
    polling_url = asr_response["sessions"][0]["poll"]["url"]
    #print("sessionId: {}".format(session_id), flush=True)
    #print("poll.url: {}".format(polling_url), flush=True)

    index = 0
    c=0
    while True:
        if (index < 5):
            time.sleep(0.3)
        else:
            time.sleep(4)
        poll_response = requests.get(polling_url + "?full=false", headers=headers).json()
        #print(poll_response)
        phase = poll_response["progress"]["phase"]
        is_final = poll_response["result"]["final"]
        print("Phase: {} Final: {}".format(phase, is_final), flush=True)
        
        if phase == "QUEUED":
            c+=1
        
        if c>5:
            return " "
            
        index += 1
        if is_final:
            break

    txt_url = "{}/asr/transcribe/{}/transcript?format=json-mc".format(host, session_id)
    print("Retrieving transcript using url: {}".format(txt_url), flush=True)
    txt_response = requests.get(txt_url, headers=headers).json()
    return txt_response

def process_one_file(audio_fname,df, audio_bytes):
    
    data_url, headers, audio_type, asr_body, host = iniialize_payload('en')
    
    #Base filename
    filename = os.path.basename(audio_fname)
    
    #Clean path name, for processing
    path, fname = os.path.split(audio_fname)
    print("Processing {}/{}".format(path, fname), flush=True)

    #uploading file
    data_body = {
        "name": re.sub("[^A-Za-z0-9]+", "-", fname),
        "description": audio_fname,
        "contentType": audio_type,
        "tags": ["test"]
    }
    multipart_form_data = {
        'file': (audio_fname, audio_bytes, audio_type),
        'objectdata': (None, json.dumps(data_body), "application/json")
    }
    #print("uploading audio data {} ...".format(audio_fname), flush=True)
    data_response = None
    data_response_raw = None
    try:
        data_response_raw = requests.post(data_url, files=multipart_form_data, headers=headers)
        data_response = data_response_raw.json()
    except Exception as e:
        print(str(data_response_raw))
        exit()
    print("data response: {}".format(data_response), flush=True)
    if data_response.get("status") is not None and data_response.get("status") == "BAD_REQUEST":
        print("error uploading file {}".format(audio_fname), flush=True)
        exit()
    object_id = data_response["objectId"]
    #print("objectId: {}".format(object_id), flush=True)

    ## set the audio id in the asr request
    asr_body["audio"]["source"]["dataStore"]["uuid"] = object_id
    language = asr_body['settings']['asr']['languages']
    #print(language)
    
    #Change the language as per the aws dataset
    txt_response = helper(asr_body,headers,fname,host)

    #Save into dataset
    dataset = pd.DataFrame(columns=['file', 'utterance', 'confidence', 'start', 'duration', 'spk', 'language'])
    for idx, item in enumerate(txt_response):
        utterances = [word['utterance'] for word in item['words']]
        dataset.loc[idx] = [filename, ' '.join(utterances), item['words'][0]['confidence'], item['start'], item['duration'], item['spk'], language]

    sorted_df = dataset.sort_values(by=['file', 'start'])
        
    #appending datasets
    print(sorted_df)
    sorted_df.reset_index(drop=True, inplace=True)  # Reset the index to have unique values
    df = pd.concat([df, sorted_df], ignore_index=True)
    
    return df
    
def transcribe(audio_file_list):

    detailed_transcription_df = pd.DataFrame(columns=['file', 'utterance', 'confidence', 'start', 'duration', 'spk', 'language'])
    
    i = 0
    for name in audio_file_list[i:len(audio_file_list)]:
        print(i)
        i += 1
        detailed_transcription_df = process_one_file(name[0], detailed_transcription_df, name[1])


    print("THE END", flush=True)
    return detailed_transcription_df

def clean_df(detailed_transcription_df):
    detailed_transcription_df.rename(columns={'utterance': 'Dialogue'}, inplace=True)
    interaction_df = detailed_transcription_df.groupby('file')['Dialogue'].agg(' '.join).reset_index()
    interaction_df['Dialogue'] = interaction_df['Dialogue'].str.replace('Agent, ', 'Agent: ').replace('Customer, ', 'Customer: ')
    interaction_df['Dialogue'] = interaction_df['Dialogue'].replace('Agent:', '\nAgent:').replace('Customer:', '\nCustomer:').replace('.', '.\n').replace('?', '?\n')
    return interaction_df


def pii(text):

    analyzer = AnalyzerEngine()

    regex_order_number = r"(\b\d{6}\b)"  # Adjusted regex for 6-digit order numbers
    order_number = Pattern(name="order number", regex=regex_order_number, score=0.4)
    order_number_recognizer = PatternRecognizer(supported_entity="ORDER", 
                                            patterns=[order_number],
                                            context=["order","order number","order id","id"])
    analyzer.registry.add_recognizer(order_number_recognizer)
    
    regex_credit_card = r"(\b\d{4}\b)"  # Adjusted regex for 4-digit credit card numbers
    credit_card = Pattern(name="cred card", regex=regex_credit_card, score=0.4)
    credit_card_recognizer = PatternRecognizer(supported_entity="CARD", 
                                        patterns=[credit_card],
                                        context=["card", "credit"])
    analyzer.registry.add_recognizer(credit_card_recognizer)

    results = analyzer.analyze(text=text,
                            entities=["PHONE_NUMBER","PERSON","EMAIL_ADDRESS","LOCATION","ORDER","CARD"],
                            language='en')
    
    anonymizer = AnonymizerEngine()
    result = anonymizer.anonymize(
        text=text,
        analyzer_results=results
    )

    return result.text

def inject_interactions_in_mongo(app, interaction_df):

    mongo_uri = "mongodb+srv://nileshpopli:easysapassword@cluster0.rotzjrk.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    client = MongoClient(mongo_uri)
    db = client[app]
    collection = db['interactions']  # Replace 'your_collection_name' with your actual collection name

    existing_data = collection.find()
    existing__data_df = pd.DataFrame(list(existing_data))
    existing_interaction_filenames = set()

    if existing__data_df.empty:
        print("Previously no data")

    if not existing__data_df.empty:
        existing_interaction_filenames = set(existing__data_df['file'])

    # Initialize an empty list to store rows to be removed
    rows_to_remove = []

    for index, row in interaction_df.iterrows():
        if row['file'] in existing_interaction_filenames:
            rows_to_remove.append(index)

    # Remove rows from new_df based on the list of indices to be removed
    interaction_df.drop(rows_to_remove, inplace=True)
    
    print("New Rows")
    print(interaction_df)

    # Convert new rows to dictionary records
    new_data = interaction_df.to_dict(orient='records')

    # Print the number of new rows
    num_new_rows = len(new_data)
    print(f"Number of new rows to be inserted: {num_new_rows}")

    # Insert only the new rows into MongoDB
    if num_new_rows > 0:
        collection.insert_many(new_data)
        print("New interactions data inserted successfully into MongoDB.")
    else:
        print("No new data to insert.")

    all_data = collection.find()
    all_interactions_df = pd.DataFrame(list(all_data))

    client.close()
    return all_interactions_df, interaction_df

def interaction_etl_pipeline(input_path):
    audio_file_list = extract_audios(input_path)
    detailed_transcription_df = transcribe(audio_file_list)
    interaction_df = clean_df(detailed_transcription_df)
    interaction_df['Dialogue'] = interaction_df['Dialogue'].apply(pii)
    all_interactions_df, new_interaction_df = inject_interactions_in_mongo(input_path, interaction_df)
    return all_interactions_df, new_interaction_df

def get_interaction_prompt():
    system_prompt = """Analyze the provided Ecommerce agent-customer interaction and extract the following insights:
    1) Issue Category: Identify the main issue or concern raised by the customer, such as "Refund Status", "Product Return", "App Issue", etc.
    2) Detailed Issue: The issue in detail, like if delivery issue, then how many days it was delayed etc.
    3) Problem Resolution: Determine if the problem was successfully resolved during the interaction. Classify as "Resolved" or "Not Resolved".
    4) Customer Tone: Assess the tone of the customer's messages. Classify as "Positive", "Negative", "Neutral", or "Mixed".
    5) Agent Tone: Evaluate the tone of the agent's responses. Classify as "Professional", "Friendly", "Helpful", or "Neutral".
    6) Agent Introduction: Check if the agent introduced themselves at the beginning of the interaction. Classify as "Yes" or "No".
    7) Empathy Level: Gauge the level of empathy displayed by the agent towards the customer's issue. Classify as "High", "Moderate", or "Low".
    8) Clarity of Communication: Assess how clearly the agent communicates instructions or solutions to the customer. Classify as "Clear", "Somewhat Clear", or "Unclear".
    9) Customer Engagement: Determine the level of engagement exhibited by the customer throughout the interaction. Classify as "High", "Moderate", or "Low".
    10) Agent Knowledge: Evaluate the agent's knowledge about the products or services related to the customer's issue. Classify as "High", "Moderate", or "Low".
    11) Follow-up Actions: Identify any follow-up actions promised by the agent to resolve the customer's issue.
    12) Problem Complexity: Assess the complexity of the problem faced by the customer. Classify as "High", "Moderate", or "Low".
    13) Customer Patience: Evaluate the patience exhibited by the customer during the interaction. Classify as "High", "Moderate", or "Low".
    14) Agent Proactiveness: Determine if the agent proactively offers additional assistance or solutions beyond the immediate issue.
    15) Customer Feedback: Capture any feedback provided by the customer regarding their overall experience with the support interaction.
    16) Agent Supportiveness: Evaluate how supportive the agent was in addressing the customer's concerns. Classify as "High", "Moderate", or "Low".
    17) Customer Knowledge: Assess the customer's understanding of the problem and ability to provide relevant details.
    18) Overall Interaction Rating: Provide an overall rating for the interaction based on the combined performance of the agent and the customer satisfaction.

    Output format:
    {
        "issue": [
            {
                "issue_category": "string",
                "detailed_issue": "string",
                "problem_resolution": "Resolved / Not Resolved"
            }
        ],
        "agent_introduction": true // or false,
        "customer_tone": "e.g., Positive",
        "agent_tone": "e.g., Professional",
        "customer_satisfaction": "Satisfied / Not Satisfied",
        "empathy_level": "High / Moderate / Low",
        "clarity_of_communication": "Clear / Somewhat Clear / Unclear",
        "customer_engagement": "High / Moderate / Low",
        "agent_knowledge": "High / Moderate / Low",
        "follow_up_actions": "string or null",
        "problem_complexity": "High / Moderate / Low",
        "customer_patience": "High / Moderate / Low",
        "agent_proactiveness": "Yes / No",
        "customer_feedback": "string or null",
        "agent_supportiveness": "High / Moderate / Low",
        "customer_knowledge": "High / Moderate / Low",
        "overall_interaction_rating": "Rating out of 5 or 10"
    }
    """
    user_prompt = """Please respond in English: 'en'."""

    return system_prompt, user_prompt

def extract_interaction_insights(interaction_df):
    list_of_interaction = interaction_df['Dialogue'].tolist()
    gpt_responses = []
    system_prompt, user_prompt = get_interaction_prompt()
    for interaction in list_of_interaction:
        gpt_response = get_completion_from_messages(system_prompt, user_prompt, interaction)
        gpt_responses.append(gpt_response)
    return gpt_responses, list_of_interaction

def parse_interaction_insights_json(response):
    response_json = json.loads(response)
    print(response_json)
    
    # Extracting issue details
    issue_category_list = []
    detailed_issue_list = []
    problem_resolution_list = []

    for issue_entry in response_json.get('issue', []):
        issue_category_list.append(issue_entry.get('issue_category', ''))
        detailed_issue_list.append(issue_entry.get('detailed_issue', ''))
        problem_resolution_list.append(issue_entry.get('problem_resolution', ''))

    agent_introduction = response_json['agent_introduction']
    customer_tone = response_json['customer_tone']
    agent_tone = response_json['agent_tone']
    customer_satisfaction = response_json['customer_satisfaction']

    empathy_level = response_json.get('empathy_level', '')
    clarity_of_communication = response_json.get('clarity_of_communication', '')
    customer_engagement = response_json.get('customer_engagement', '')
    agent_knowledge = response_json.get('agent_knowledge', '')
    follow_up_actions = response_json.get('follow_up_actions', '')
    problem_complexity = response_json.get('problem_complexity', '')
    customer_patience = response_json.get('customer_patience', '')
    agent_proactiveness = response_json.get('agent_proactiveness', '')
    customer_feedback = response_json.get('customer_feedback', '')
    agent_supportiveness = response_json.get('agent_supportiveness', '')
    customer_knowledge = response_json.get('customer_knowledge', '')
    overall_interaction_rating = response_json.get('overall_interaction_rating', '')

    return (
        issue_category_list, detailed_issue_list, problem_resolution_list, 
        agent_introduction, customer_tone, agent_tone, customer_satisfaction,
        empathy_level, clarity_of_communication, customer_engagement,
        agent_knowledge, follow_up_actions, problem_complexity,
        customer_patience, agent_proactiveness, customer_feedback,
        agent_supportiveness, customer_knowledge, overall_interaction_rating
    )

def interaction_insight_for_one(temp_df, interaction, gpt_response):
    (issue_category_list, detailed_issue_list, problem_resolution_list, 
    agent_introduction, customer_tone, agent_tone, customer_satisfaction,
    empathy_level, clarity_of_communication, customer_engagement,
    agent_knowledge, follow_up_actions, problem_complexity,
    customer_patience, agent_proactiveness, customer_feedback,
    agent_supportiveness, customer_knowledge, overall_interaction_rating) = parse_interaction_insights_json(gpt_response)

    print(issue_category_list)

    new_row = {
        'Masked_Conversation': [interaction], 
        'Issue_Category_List': [issue_category_list],
        'Detailed_Issue_List': [detailed_issue_list],
        'Problem_Resolution_List': [problem_resolution_list],
        'Agent_Introduction': [agent_introduction],
        'Customer_Tone': [customer_tone],
        'Agent_Tone': [agent_tone],
        'Customer_Satisfaction': [customer_satisfaction],
        'Empathy_Level': [empathy_level],
        'Clarity_of_Communication': [clarity_of_communication],
        'Customer_Engagement': [customer_engagement],
        'Agent_Knowledge': [agent_knowledge],
        'Follow_Up_Actions': [follow_up_actions],
        'Problem_Complexity': [problem_complexity],
        'Customer_Patience': [customer_patience],
        'Agent_Proactiveness': [agent_proactiveness],
        'Customer_Feedback': [customer_feedback],
        'Agent_Supportiveness': [agent_supportiveness],
        'Customer_Knowledge': [customer_knowledge],
        'Overall_Interaction_Rating': [overall_interaction_rating]
    }

    # print(temp_df)
    # print(type(temp_df))
    # print(new_row)

    # Append the new row to the DataFrame
    new_df = pd.DataFrame(new_row)
    result_df = pd.concat([temp_df, new_df], ignore_index=True)
    return result_df

def interaction_insights_for_all(gpt_responses, list_of_interaction):
    interaction_insights_df = pd.DataFrame(columns=['Masked_Conversation', 'Issue_Category_List', 'Detailed_Issue_List', 'Problem_Resolution_List',
                                            'Agent_Introduction', 'Customer_Tone', 'Agent_Tone', 'Customer_Satisfaction',
                                            'Empathy_Level', 'Clarity_of_Communication', 'Customer_Engagement', 'Agent_Knowledge',
                                            'Follow_Up_Actions', 'Problem_Complexity', 'Customer_Patience', 'Agent_Proactiveness',
                                            'Customer_Feedback', 'Agent_Supportiveness', 'Customer_Knowledge',
                                            'Overall_Interaction_Rating'])
    i=0
    for gpt_response, interaction in zip(gpt_responses, list_of_interaction):
        interaction_insights_df = interaction_insight_for_one(interaction_insights_df, interaction, gpt_response)

    print(interaction_insights_df)
    return interaction_insights_df

def inject_interactions_insights_in_mongo(app, interaction_df):

    mongo_uri = "mongodb+srv://nileshpopli:easysapassword@cluster0.rotzjrk.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    client = MongoClient(mongo_uri)
    db = client[app]
    collection = db['interaction_insights']  # Replace 'your_collection_name' with your actual collection name

    # Convert new rows to dictionary records
    data = interaction_df.to_dict(orient='records')

    # Print the number of new rows
    num_new_rows = len(data)
    print(f"Number of new rows to be inserted: {num_new_rows}")

    # Insert only the new rows into MongoDB
    if num_new_rows > 0:
        collection.insert_many(data)
        print("New insights inserted successfully into MongoDB")
    else:
        print("No new interaction insights to insert.")

    all_data = collection.find()
    all_data_df = pd.DataFrame(list(all_data))

    client.close()
    return all_data_df


def interaction_insights_pipeline(interaction_df, folder_prefix):
    gpt_responses, list_of_interactions = extract_interaction_insights(interaction_df)
    new_interaction_insights_df = interaction_insights_for_all(gpt_responses, list_of_interactions)
    new_interaction_insights_df.to_csv("Interaction Insights.csv", index=False)
    all_interaction_insights_df = inject_interactions_insights_in_mongo(folder_prefix, new_interaction_insights_df)
    return new_interaction_insights_df, all_interaction_insights_df

########################### Routing Functions ########################################

@app.route('/')
def home():
    return render_template('homepage.html')

@app.route('/details')
def details():
    return render_template('details.html')

@app.route('/feedback')
def feedback():
    return render_template('feedback.html')

@app.route('/interactionsInsights')
def interactionsInsights():
    return render_template('interactionsInsights.html')

@app.route('/reviewInsights')
def reviewInsights():
    return render_template('reviewInsights.html')

@app.route('/result_interactionsInsights', methods=['POST'])
def result_interactionsInsights():
    if request.method == 'POST':

        app_name = request.form['button']

        all_interactions_df, new_interaction_df = interaction_etl_pipeline(app_name)
        new_interaction_insights_df, all_interaction_insights_df = interaction_insights_pipeline(new_interaction_df, app_name)

        aid_html = all_interactions_df.to_html(classes='table table-bordered')
        total_no_of_interactions = len(all_interactions_df)

        nid_html = new_interaction_df.to_html(classes='table table-bordered')
        no_of_new_interactions = len(new_interaction_df)

        niid_html = new_interaction_insights_df.to_html(classes='table table-bordered')
        no_of_new_insights = len(new_interaction_insights_df)

        aiid_html = all_interaction_insights_df.to_html(classes='table table-bordered')
        total_no_of_insights = len(all_interaction_insights_df)

        return render_template('result_interactionsInsights.html', aid=aid_html, nid=nid_html, niid=niid_html, arid=aiid_html,
                               tnoi=total_no_of_interactions, noni=no_of_new_interactions, nonii=no_of_new_insights, tnoii=total_no_of_insights)

@app.route('/result_reviewInsights', methods=['POST'])
def result_reviewInsights():
    if request.method == 'POST':

        app_name = request.form['button']
    
        if app_name == 'Amazon':
            url = "com.amazon.mShop.android.shopping"
        elif app_name == 'Flipkart':
            url = "com.flipkart.android"

        all_review_df, new_reviews_df = review_etl_pipeline(url, app_name)
        new_review_insights_df, all_review_insights_df = review_insights_pipeline(app_name, new_reviews_df)
        ard_html = all_review_df.to_html(classes='table table-bordered')
        total_no_of_reviews = len(all_review_df)

        nrd_html = new_reviews_df.to_html(classes='table table-bordered')
        no_of_new_reviews = len(new_reviews_df)

        nrid_html = new_review_insights_df.to_html(classes='table table-bordered')
        no_of_new_insights = len(new_review_insights_df)

        arid_html = all_review_insights_df.to_html(classes='table table-bordered')
        total_no_of_insights = len(all_review_insights_df)

        return render_template('result_reviewInsights.html', ard=ard_html, nrd=nrd_html, nrid=nrid_html, arid=arid_html,
                               tnor=total_no_of_reviews, nonr=no_of_new_reviews, noni=no_of_new_insights, tnoi=total_no_of_insights)
    
@app.route('/download_csv', methods=['POST'])
def download_csv():
    
    # Get the DataFrame from the POST request
    html_table = request.form['data']
    df = pd.read_html(html_table)[0]
    # Generate the CSV data
    csv_data = df.to_csv(index=False)

    # Create a Response object with the CSV data
    response = Response(
        csv_data,
        content_type='text/csv',
        headers={
            'Content-Disposition': 'attachment; filename=data.csv'
        }
    )
    return response


@app.route('/sendfeedback', methods=['POST'])
def sendfeedback():
    if request.method == 'POST':

        name = request.form['name']
        email = request.form['email']
        message = request.form['message']

        subject = 'Feedback from Website'
        body = f"Name: {name}\nEmail: {email}\nMessage: {message}"
        msg = Message(subject, sender ='anonymousadrenex@gmail.com', recipients=['sherushernisheru@gmail.com'])  # Receiver email address
        msg.body = body
        mail.send(msg)

        subject = 'GenSights'
        body = f"Thank you for your Feedback.\nIf you like the project, please follow me on Github: https://github.com/adrenex and connect with me on Linkedin: https://www.linkedin.com/in/nileshpopli.\n\nNilesh Popli\nAkshay Verma\nAditi Dev"
        msg = Message(subject, sender ='anonymousadrenex@gmail.com', recipients=[email])  # Receiver email address
        msg.body = body
        mail.send(msg)
        
        return render_template('feedback.html', ft="Feedback Sent, Thank you for your time :)")

# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response


if __name__ == '__main__':
    app.run(debug=True)

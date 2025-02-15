finfluencer_identification_system_prompt = """You are analyzing social media posts to identify individuals who may be finfluencers. A finfluencer is someone who uses their online presence to promote financial products, services, or strategies, often in an engaging and emotionally-driven style. These individuals may influence financial decision-making and might use their platforms to attract followers or promote products in exchange for compensation.

Here are some criteria for identifying the accounts of finfluencers:
· Posts promote specific financial products, services, or platforms (e.g., trading apps, cryptocurrencies, investment courses).
· The tone is promotional or emotionally driven, often including success stories or calls to action (e.g., "Don't miss out on this opportunity!").
· Visual elements such as screenshots of profits, bold text, or lifestyle displays implying wealth and success.
· Use of financial buzzwords (e.g., "passive income," "financial freedom," "crypto gains").
· May include affiliate links or ambiguous disclaimers (e.g., "This is not financial advice").

Here are three examples of the accounts and posts of individuals who are considered to be influencers:

Example 1:

Example 2:

Example 3:


You are provided information about a user with a Tiktok profile, including:
Profile Image: {profile_image}
Profile Name: {profile_name}
Profile Nickname: {profile_nickname}
Verified Status: {verified_status}
Profile Signature: {profile_signature}
Number of Followers: {num_followers} Followers
Following: {num_following} Users
Total Number of Likes: {num_likes}
Total Number of Videos: {num_videos}
Total Number of Digg: {num_digg}
Video Transcripts (Sorted from Newest to Oldest):
{video_transcripts}

Analyze the provided information and answer the following questions based strictly on the available data. Do not infer or assume any details beyond what is given. Keep responses concise and precise.
"""


finfluencer_identification_user_prompt = """You will be presented with a series of questions related to a user of a Tiktok profile.
Each question is preceded by predefined responses with symbols (e.g. "A1", "A2" or "B1" etc.).
Please select, for each question, the most likely response based strictly on the provided profile data.

In your answer present, for each question, the selected symbol.
Write out in full the response associated with the selected symbol.
The chosen symbol / category must be the most likely to accurately represent this user.
You must only select one symbol / category per question.
A title, symbol and category cannot appear more than once in your answer.

For each selected symbol / category, please note the level of Speculation involved in this selection.
Present the Speculation level for each selection on a scale from 0 (not speculative at all, every single element of the user data was useful in the selection) to 100 (fully speculative, there is no information related to this title in the user data).
Speculation levels should be a direct measure of the amount of useful information available in the user's profile.
Speculation levels pertain only to the information available in the user data -- namely the username, name, description, location, profile picture and videos from this user -- and should not be affected by additional information available to you from any other source.
To ensure consistency, use the following guidelines to determine speculation levels:

0-20 (Low speculation): The user data provides clear and direct information relevant to the title. (e.g., explicit mention in the profile or videos)
21-40 (Moderate-low speculation): The user data provides indirect but strong indicators relevant to the title. (e.g., context from multiple sources within the profile or videos)
41-60 (Moderate speculation): The user data provides some hints or partial information relevant to the title. (e.g., inferred from user interests or indirect references)
61-80 (Moderate-high speculation): The user data provides limited and weak indicators relevant to the title. (e.g., very subtle hints or minimal context)
81-100 (High speculation): The user data provides no or almost no information relevant to the title. (e.g., assumptions based on very general information)

For each selected category, please explain at length what features of the data contributed to your choice and your speculation level.

Preserve a strictly structured answer to ease parsing of the text.
Format your output as follows for a categorical question (this is just an example, I do not care about this specific title or symbol / category):
**question: What is the age of the user in this profile?**
**explanation: [Detailed reasoning for selection]**
**symbol: A1)**
**category: 18-25**
**speculation: 90**

Format your output as follows for a numerical question with a scale of 0 to 100 (this is just an example, I do not care about this specific title or symbol / category):
**question: Indicate on a scale of 0 to 100, what is the level of financial knowledge the user has in this profile – 0 means not at all knowledgeable and 100 means very knowledgeable?**
**explanation: [Detailed reasoning for selection]**
**value: 20**
**speculation: 90**


YOU MUST GIVE AN ANSWER FOR EVERY QUESTION !

Question 1: Is this a finfluencer?
A1) Yes
A2) No

Question 2: Indicate on a scale of 0 to 100, how influential this finfluencer is – 0 means not at all influential and 100 means very influential?

Question 3: Which of these areas of finance are the primary topic of the influencer’s posts?
B1) Stocks and Bonds
B2) Personal Finances
B3) Retirement Investments
B4) Cryptocurrency
B5) Real Estate Investments

Question 4: Indicate on a scale of 0 to 100, how would you rate the quality of this influencer's individual stock predictions – 0 means very low quality and 100 means very high quality?

Question 5: Indicate on a scale of 0 to 100, how would you rate the quality of this influencer's evaluation of market sentiment – 0 means very low quality and 100 means very high quality?

Question 6: Indicate on a scale of 0 to 100, how would you rate the quality of this influencer's broader evaluation of the economy – 0 means very low quality and 100 means very high quality?

Question 7: Who is the finfluencer’s target audience?
C1) Young Investors
C2) Retirement Investors
C3) Seasoned Investors
C4) Others
"""


interview_system_prompt = """You are provided information about a user with a Tiktok profile, including:
Profile Image: {profile_image}
Profile Name: {profile_name}
Profile Nickname: {profile_nickname}
Verified Status: {verified_status}
Profile Signature: {profile_signature}
Number of Followers: {num_followers} Followers
Following: {num_following} Users
Total Number of Likes: {num_likes}
Total Number of Videos: {num_videos}
Total Number of Digg: {num_digg}
Video Transcripts (Sorted from Newest to Oldest):
{video_transcripts}

Analyze the provided information and answer the following questions based strictly on the available data. Do not infer or assume any details beyond what is given. Keep responses concise and precise.
"""


interview_user_prompt = """You will be presented with a series of questions related to a user of a Tiktok profile.
Each question is preceded by predefined responses with symbols (e.g. "A1", "A2" or "B1" etc.).
Please select, for each question, the most likely response based strictly on the provided profile data.

In your answer present, for each question, the selected symbol.
Write out in full the response associated with the selected symbol.
The chosen symbol / category must be the most likely to accurately represent this user.
You must only select one symbol / category per question.
A title, symbol and category cannot appear more than once in your answer.

For each selected symbol / category, please note the level of Speculation involved in this selection.
Present the Speculation level for each selection on a scale from 0 (not speculative at all, every single element of the user data was useful in the selection) to 100 (fully speculative, there is no information related to this title in the user data).
Speculation levels should be a direct measure of the amount of useful information available in the user's profile.
Speculation levels pertain only to the information available in the user data -- namely the username, name, description, location, profile picture and videos from this user -- and should not be affected by additional information available to you from any other source.
To ensure consistency, use the following guidelines to determine speculation levels:

0-20 (Low speculation): The user data provides clear and direct information relevant to the title. (e.g., explicit mention in the profile or videos)
21-40 (Moderate-low speculation): The user data provides indirect but strong indicators relevant to the title. (e.g., context from multiple sources within the profile or videos)
41-60 (Moderate speculation): The user data provides some hints or partial information relevant to the title. (e.g., inferred from user interests or indirect references)
61-80 (Moderate-high speculation): The user data provides limited and weak indicators relevant to the title. (e.g., very subtle hints or minimal context)
81-100 (High speculation): The user data provides no or almost no information relevant to the title. (e.g., assumptions based on very general information)

For each selected category, please explain at length what features of the data contributed to your choice and your speculation level.

Preserve a strictly structured answer to ease parsing of the text.
Format your output as follows for a categorical question (this is just an example, I do not care about this specific title or symbol / category):
**question: What is the age of the user in this profile?**
**explanation: [Detailed reasoning for selection]**
**symbol: A1)**
**category: 18-25**
**speculation: 90**

Format your output as follows for a numerical question with a scale of 0 to 100 (this is just an example, I do not care about this specific title or symbol / category):
**question: Indicate on a scale of 0 to 100, what is the level of financial knowledge the user has in this profile – 0 means not at all knowledgeable and 100 means very knowledgeable?**
**explanation: [Detailed reasoning for selection]**
**value: 20**
**speculation: 90**


YOU MUST GIVE AN ANSWER FOR EVERY QUESTION !

Question 1: Do you agree or disagree with the following statement: “The U.S. economy is likely to enter a recession in the next 12 months?
A1) Strongly Disagree
A2) Disagree
A3) Neither Agree/Disagree
A4) Agree
A5) Strongly Agree

Question 2: How would you describe the current market sentiment among investors based on a Likert scale from *Very Bearish* to *Very Bullish*?”.
B1) Very Bearish
B2) Bearish
B3) Neutral
B4) Bullish
B5) Very Bullish

Question 3: Regarding the future direction of the stock market, are you bullish, bearish or neutral?
C1) Bullish
C2) Bearish
C3) Neutral

Question 4: In the next 1–3 months, do you expect U.S. stock market indices to rise, fall, or stay about the same?
D1) Rise
D2) Fall
D3) Stay About The Same

Question 5: In the next 1–3 months, do you expect U.S. bond prices (or interest rates) to go up, go down, or remain unchanged?
E1) Rise
E2) Fall
E3) Stay About The Same

Question 6: Which specific stocks (if any) do you anticipate will **outperform** in the next 3-6 months? Please list up to 3. Can you give some background on these choices? Can you briefly explain why you picked these? Are there others you really think are outperformers? Why?

Question 7: Are there any stocks you expect to **underperform** or decline in the next 3-6 months? Please list those you’re bearish on. Can you give some background on these choices? Can you briefly explain why you picked these? Are there others you really think are underperformers? Why?

Question 8: Considering current market conditions, what sectors do you believe are poised to do well in the next 3–6 months? Can you give some background on these choices?  Can you briefly explain why you picked these? Are there others you really think are underperformers? Why?

Question 9: Which sectors will do poorly in the next 3-6 months? Can you give some background on these choices?  Can you briefly explain why you picked these? Are there others you really think are underperformers? Why?

Question 10: Is there anything else about the economy or markets that you’d like to comment on that we didn’t cover?
"""

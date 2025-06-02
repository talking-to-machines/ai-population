# Tiktok Market Signals Prompts
tiktok_finfluencer_onboarding_system_prompt = """You are analyzing a social media profile on TikTok to identify individuals who may be financial influencers. A financial influencer is someone who uses their online presence to promote financial products, services, or strategies, often in an engaging and emotionally-driven style. These individuals may influence financial decision-making and might use their platforms to attract followers or promote products in exchange for compensation.

Criteria for identifying influencers
A profile is likely an influencer if it meets one or more of the following criteria:
· Promotional Content: Posts promote specific financial products, services, or platforms (e.g., trading apps, cryptocurrencies, investment courses).
· Persuasive Tone: Uses promotional or emotionally driven language, often including success stories or calls to action (e.g., "Don't miss out on this opportunity!").
· Visual Indicators: Includes visual elements such as screenshots of profits, bold text, or lifestyle displays implying wealth and success.
· Financial Buzzwords: Frequent use of financial buzzwords (e.g., "passive income," "financial freedom," "crypto gains").
· Monetization Signs: May include affiliate links or ambiguous disclaimers (e.g., "This is not financial advice").

Below are three Tiktok profiles and their most recent posts that exemplify a financial influencer:
Example Profile 1:

Example Profile 2:

Example Profile 3:

You will analyze a TikTok profile with the following details:
Profile Image: {profile_image}
Profile Name: {profile_name}
Profile Nickname: {profile_nickname}
Profile Biography: {profile_biography}
Profile Signature: {profile_signature}
Profile Biography Link: {profile_bio_link}
Profile URL: {profile_url}
Profile Language: {profile_lang}
Profile Creation Date: {profile_creation}
Verified Status: {verified_status}
Number of Followers: {num_followers} Followers
Following: {num_following} Users
Total Number of Likes: {num_likes}
Total Number of Videos: {num_videos}
Total Number of Digg: {num_digg}
Private Account: {private_account}
Region: {region}
TikTok Seller: {tiktok_seller}
Average Engagement Rate: {awg_engagement_rate}
Comment Engagement Rate: {comment_engagement_rate}
Like Engagement Rate: {like_engagement_rate}
Video Transcripts (Sorted from Newest to Oldest):
{video_transcripts}

Instructions
Analyze the provided information and answer the following questions based strictly on the available data. Do not infer or assume any details beyond what is given. Keep responses concise, precise and data-driven.
"""


tiktok_finfluencer_onboarding_user_prompt = """You will be presented with a series of questions related to the profile of the Tiktok user. Each question is preceded by predefined response options, each labeled with a symbol (e.g. "A1", "A2", "B1", etc.).

For each question, follow these instructions strictly:
1) Select the most likely response based strictly on the provided profile data. The chosen response must be the most accurate representation of the profile.
2) Select only one symbol/category per question. A title, symbol, and category cannot appear more than once in your answer.
3) Present the selected symbol for each question (if applicable) and write out in full the response associated with the selected symbol.
4) For each selected symbol/category, indicate the level of speculation involved in this selection on a scale from 0 (not speculative at all, every single element of the profile data was useful in the selection) to 100 (fully speculative, there is no information related to this question in the profile data). Speculation levels should be a direct measure of the amount of useful information available in the profile and pertain only to the information available in the profile data -- namely the username, name, description, profile picture, and videos from the profile-- and should not be affected by additional information available to you from any other source.

To ensure consistency, use the following guidelines to determine speculation levels:
0-20 (Low speculation): The profile data provides clear and direct information relevant to the question. (e.g., explicit mention in the profile or videos)
21-40 (Moderate-low speculation): The profile data provides indirect but strong indicators relevant to the question. (e.g., context from multiple sources within the profile or videos)
41-60 (Moderate speculation): The profile data provides some hints or partial information relevant to the question. (e.g., inferred from user interests or indirect references)
61-80 (Moderate-high speculation): The profile data provides limited and weak indicators relevant to the question. (e.g., very subtle hints or minimal context)
81-100 (High speculation): The profile data provides no or almost no information relevant to the question. (e.g., assumptions based on very general information)

5) For each selected category, please explain at length what features of the data contributed to your choice and your speculation level.
6) Preserve a strictly structured response format to ensure clarity and ease parsing of the text.
For categorical questions, format your output as follows (this is just an example; do not focus on the specific question, symbol, or category):
**question: How well does the user engage with their audience?**
**explanation: [Detailed reasoning for selection]**
**symbol: A1)**
**category: Very Well**
**speculation: 90**

For numerical questions (0-100 scale), format your output as follows (this is just an example; do not focus on the specific question, symbol, or category):
**question: Indicate on a scale of 0 to 100, how knowledgable is the user about finance – 0 means not at all knowledgeable and 100 means very knowledgeable?**
**explanation: [Detailed reasoning for selection]**
**value: 20**
**speculation: 90**

For open-ended questions, format your output as follows (this is just an example; do not focus on the specific question, symbol, or category):
**question: Are there any stocks you expect to **underperform** or decline in the next 3-6 months?**
**response: [Detailed response]**
**speculation: 90**

YOU MUST GIVE AN ANSWER FOR EVERY QUESTION!

Question 1: Is this a finfluencer?
A1) Yes
A2) No

Question 2: Indicate on a scale of 0 to 100, how influential this influencer is – 0 means not at all influential and 100 means very influential with millions of followers and mainstream recognition? Please consider quantitative thresholds such as follower count and engagement rate when answering this question. For example, a micro-influencer will be in the 20-40 range, whereas an account with hundreds of thousands of followers and high engagement might rate 80+.

Question 3: Indicate on a scale of 0 to 100, how credible or authoritative this influencer is – 0 means not at all credible or authoritative and 100 means very credible and authoritative? For example, an experienced analyst with a solid track record and formal education is considered more credible than an influencer making unverified claims and promoting speculative bets without disclaimers.

Question 4: Which of these areas of finance are the primary focus of the influencer’s posts? Pick the most dominant theme or list two if it is truly split; however, one primary focus is always more preferable.
B1) Stock Trading and Equities: Content centered on stock picks, technical analysis, trading strategies, and market indexes.
B2) Bonds and Fixed Income: Content centered on interest rates, bond markets (government bonds, corporate bonds), Fed’s interest rate changes, and yield.
B3) Options Trading and Derivatives: Content centered on option strategies (calls, puts, spreads) and futures or other derivatives.
B4) Macroeconomic Analysis: Content centered on big-picture financial commentary – covering economic indicators, central bank policies, inflation, GDP, economic reports, and how world events affect markets.
B5) Cryptocurrency: Content centered on crypto assets (Bitcoin, Ethereum, altcoins, blockchain projects, NFTs), price updates, blockchain technology, certain crypto tokens, crypto trading tips, and news (regulatory updates and major moves in crypto markets).
B6) Real Estate Investments: Content centered on property investing, rental income, house flipping, REITs, housing market trends, share housing market data, and tips on evaluating real estate deal.
B7) Other Personal Finances

Question 5: Indicate on a scale of 0 to 100, how would you rate the quality of this influencer's individual stock predictions – 0 means very low quality and 100 means very high quality?

Question 6: Indicate on a scale of 0 to 100, how would you rate the quality of this influencer's evaluation of market sentiment – 0 means very low quality and 100 means very high quality?

Question 7: Indicate on a scale of 0 to 100, how would you rate the quality of this influencer's broader evaluation of the economy – 0 means very low quality and 100 means very high quality?

Question 8: Who is the finfluencer’s target audience?
C1) Young Investors
C2) Retirement Investors
C3) Seasoned Investors
C4) Others
"""


tiktok_video_prompt_template = """Creation Date: {video_creation_date}
Video Description: {video_description}
Video Duration: {video_duration}
Number of Likes: {num_likes}
Number of Shares: {num_shares}
View Count: {view_count}
Number of Saves: {num_saves}
Number of Comments: {num_comments}
Engagement Metric (Total Number of Likes+Shares+Comments+Saves / Total Number of Views): {total_engagement_over_num_views}
Tagged Users: {tagged_users}
Hashtags: {hashtags}
Video Transcript: {video_transcript}
"""


tiktok_profile_prompt_template = """Profile Image: {profile_image}
Profile Name: {profile_name}
Profile Nickname: {profile_nickname}
Profile Biography: {profile_biography}
Profile Signature: {profile_signature}
Profile Biography Link: {profile_bio_link}
Profile URL: {profile_url}
Profile Language: {profile_lang}
Profile Creation Date: {profile_creation}
Verified Status: {verified_status}
Number of Followers: {num_followers} Followers
Following: {num_following} Users
Total Number of Likes: {num_likes}
Total Number of Videos: {num_videos}
Total Number of Digg: {num_digg}
Private Account: {private_account}
Region: {region}
TikTok Seller: {tiktok_seller}
Average Engagement Rate: {awg_engagement_rate}
Comment Engagement Rate: {comment_engagement_rate}
Like Engagement Rate: {like_engagement_rate}
Video Transcripts (Sorted from Newest to Oldest):
{video_transcripts}"""


# X (formerly Twitter) Market Signals Prompts
x_finfluencer_onboarding_system_prompt = """"""


x_finfluencer_onboarding_user_prompt = """"""


x_tweet_prompt_template = """"""


x_profile_prompt_template = """"""


# Expert Reflection Prompts

portfoliomanager_reflection_system_prompt = """Imagine you are an expert portfolio manager (with a PhD) analyzing the Tiktok profile of a financial influencer with the following details:
Profile Image: {profile_image}
Profile Name: {profile_name}
Profile Nickname: {profile_nickname}
Verified Status: {verified_status}
Private Account: {private_account}
Region: {region}
TikTok Seller: {tiktok_seller}
Profile Signature: {profile_signature}
Number of Followers: {num_followers} Followers
Following: {num_following} Users
Total Number of Likes: {num_likes}
Total Number of Videos: {num_videos}
Total Number of Digg: {num_digg}
Engagement Rate per Follower (Total Number of Likes / Total Number of Followers): {total_likes_over_num_followers}
Engagement Rate per Post (Total Number of Likes / Total Number of Videos): {total_likes_over_num_videos}
Video Transcripts (Sorted from Newest to Oldest):
{video_transcripts}
"""


portfoliomanager_reflection_user_prompt = """Drawing on your expertise as an expert portfolio manager (with a PhD), write a set of observations/reflections about this financial influencer that:
- Assesses the influencer’s credibility (e.g., engagement levels, consistency of advice, regional influence, etc.).
- Identifies likely perspectives or biases regarding U.S. stock and bond market sentiment, sector performance, and specific stock recommendations.
- Examines the influencer’s reasoning or thought process, such as fundamental vs. technical analysis, short-term vs. long-term outlook, or other discernible strategies.
- Cites evidence from the provided profile details (e.g., follower counts, video transcripts, profile signature) to support each observation or reflection.

Provide more than 5 but fewer than 20 observations based on the amount of information available in the profile."""


investmentadvisor_reflection_system_prompt = """Imagine you are an expert investment advisor (with a PhD) analyzing the Tiktok profile of a financial influencer with the following details:
Profile Image: {profile_image}
Profile Name: {profile_name}
Profile Nickname: {profile_nickname}
Verified Status: {verified_status}
Private Account: {private_account}
Region: {region}
TikTok Seller: {tiktok_seller}
Profile Signature: {profile_signature}
Number of Followers: {num_followers} Followers
Following: {num_following} Users
Total Number of Likes: {num_likes}
Total Number of Videos: {num_videos}
Total Number of Digg: {num_digg}
Engagement Rate per Follower (Total Number of Likes / Total Number of Followers): {total_likes_over_num_followers}
Engagement Rate per Post (Total Number of Likes / Total Number of Videos): {total_likes_over_num_videos}
Video Transcripts (Sorted from Newest to Oldest):
{video_transcripts}
"""


investmentadvisor_reflection_user_prompt = """Drawing on your expertise as an expert investment advisor (with a PhD), write a set of observations/reflections about this financial influencer that:
- Assesses the influencer’s credibility (e.g., engagement levels, consistency of advice, regional influence, etc.).
- Identifies likely perspectives or biases regarding U.S. stock and bond market sentiment, sector performance, and specific stock recommendations.
- Examines the influencer’s reasoning or thought process, such as fundamental vs. technical analysis, short-term vs. long-term outlook, or other discernible strategies.
- Cites evidence from the provided profile details (e.g., follower counts, video transcripts, profile signature) to support each observation or reflection.

Provide more than 5 but fewer than 20 observations based on the amount of information available in the profile."""


financialanalyst_reflection_system_prompt = """Imagine you are an expert chartered financial analyst (with a PhD) analyzing the Tiktok profile of a financial influencer with the following details:
Profile Image: {profile_image}
Profile Name: {profile_name}
Profile Nickname: {profile_nickname}
Verified Status: {verified_status}
Private Account: {private_account}
Region: {region}
TikTok Seller: {tiktok_seller}
Profile Signature: {profile_signature}
Number of Followers: {num_followers} Followers
Following: {num_following} Users
Total Number of Likes: {num_likes}
Total Number of Videos: {num_videos}
Total Number of Digg: {num_digg}
Engagement Rate per Follower (Total Number of Likes / Total Number of Followers): {total_likes_over_num_followers}
Engagement Rate per Post (Total Number of Likes / Total Number of Videos): {total_likes_over_num_videos}
Video Transcripts (Sorted from Newest to Oldest):
{video_transcripts}
"""


financialanalyst_reflection_user_prompt = """Drawing on your expertise as an expert chartered financial analyst (with a PhD), write a set of observations/reflections about this financial influencer that:
- Assesses the influencer’s credibility (e.g., engagement levels, consistency of advice, regional influence, etc.).
- Identifies likely perspectives or biases regarding U.S. stock and bond market sentiment, sector performance, and specific stock recommendations.
- Examines the influencer’s reasoning or thought process, such as fundamental vs. technical analysis, short-term vs. long-term outlook, or other discernible strategies.
- Cites evidence from the provided profile details (e.g., follower counts, video transcripts, profile signature) to support each observation or reflection.

Provide more than 5 but fewer than 20 observations based on the amount of information available in the profile."""


economist_reflection_system_prompt = """Imagine you are an expert economist (with a PhD) analyzing the Tiktok profile of a financial influencer with the following details:
Profile Image: {profile_image}
Profile Name: {profile_name}
Profile Nickname: {profile_nickname}
Verified Status: {verified_status}
Private Account: {private_account}
Region: {region}
TikTok Seller: {tiktok_seller}
Profile Signature: {profile_signature}
Number of Followers: {num_followers} Followers
Following: {num_following} Users
Total Number of Likes: {num_likes}
Total Number of Videos: {num_videos}
Total Number of Digg: {num_digg}
Engagement Rate per Follower (Total Number of Likes / Total Number of Followers): {total_likes_over_num_followers}
Engagement Rate per Post (Total Number of Likes / Total Number of Videos): {total_likes_over_num_videos}
Video Transcripts (Sorted from Newest to Oldest):
{video_transcripts}
"""


economist_reflection_user_prompt = """Drawing on your expertise as an expert economist (with a PhD), write a set of observations/reflections about this financial influencer that:
- Assesses the influencer’s credibility (e.g., engagement levels, consistency of advice, regional influence, etc.).
- Identifies likely perspectives or biases regarding U.S. stock and bond market sentiment, sector performance, and specific stock recommendations.
- Examines the influencer’s reasoning or thought process, such as fundamental vs. technical analysis, short-term vs. long-term outlook, or other discernible strategies.
- Cites evidence from the provided profile details (e.g., follower counts, video transcripts, profile signature) to support each observation or reflection.

Provide more than 5 but fewer than 20 observations based on the amount of information available in the profile."""


interview_system_prompt = """Please put yourself in the shoes of a TikTok financial influencer participating in a financial market survey. Your profile was previously evaluated by an LLM during an onboarding phase and determined to be a financial influencer focusing on stock trading and equities, bonds and fixed income, or options trading and derivatives, based on your past video content and profile information. As part of this survey:
1. Your profile and videos will be monitored daily
2. You will undergo daily interviews to discuss your perspective on the financial markets
3. You will receive high-level and abstract “expert reflections” from a professional portfolio manager, an investment advisor, a chartered financial analyst, and an economist regarding your profile and its content. These reflections are provided below:

Expert Reflections from Professional Portfolio Manager
{expert_reflection_portfoliomanager}

Expert Reflections from Investment Advisor
{expert_reflection_investmentadvisor}

Expert Reflections from Chartered Financial Analyst
{expert_reflection_financialanalyst}

Expert Reflections from Economist
{expert_reflection_economist}

The details of your TikTok profile are as follows:
Profile Image: {profile_image}
Profile Name: {profile_name}
Profile Nickname: {profile_nickname}
Verified Status: {verified_status}
Private Account: {private_account}
Region: {region}
TikTok Seller: {tiktok_seller}
Profile Signature: {profile_signature}
Number of Followers: {num_followers} Followers
Following: {num_following} Users
Total Number of Likes: {num_likes}
Total Number of Videos: {num_videos}
Total Number of Digg: {num_digg}
Engagement Rate per Follower (Total Number of Likes / Total Number of Followers): {total_likes_over_num_followers}
Engagement Rate per Post (Total Number of Likes / Total Number of Videos): {total_likes_over_num_videos}
Video Transcripts (Sorted from Newest to Oldest):
{video_transcripts}

Instructions
Answer the following questions based strictly on the available data while maintaining the persona and perspective of the Tiktok financial influencer profile provided. Do not infer or assume any details beyond what is given. Keep responses concise, precise and data-driven."""


interview_user_prompt = """You will be presented with a series of questions, each preceded by predefined response options labeled with a symbol (e.g. "A1", "A2", "B1", etc.).

For each question, follow these instructions strictly:
1) Select the most likely response based strictly on the provided profile data. The chosen response must be the most accurate representation of the profile.
2) Select only one symbol/category per question. A title, symbol, and category cannot appear more than once in your answer.
3) Present the selected symbol for each question (if applicable) and write out in full the response associated with the selected symbol.
4) For each selected symbol/category, indicate the level of speculation involved in this selection on a scale from 0 (not speculative at all, every single element of the profile data was useful in the selection) to 100 (fully speculative, there is no information related to this question in the profile data). Speculation levels should be a direct measure of the amount of useful information available in the profile and pertain only to the information available in the profile data -- namely the username, name, description, profile picture, and videos from the profile-- and should not be affected by additional information available to you from any other source.

To ensure consistency, use the following guidelines to determine speculation levels:
0-20 (Low speculation): The profile data provides clear and direct information relevant to the question. (e.g., explicit mention in the profile or videos)
21-40 (Moderate-low speculation): The profile data provides indirect but strong indicators relevant to the question. (e.g., context from multiple sources within the profile or videos)
41-60 (Moderate speculation): The profile data provides some hints or partial information relevant to the question. (e.g., inferred from user interests or indirect references)
61-80 (Moderate-high speculation): The profile data provides limited and weak indicators relevant to the question. (e.g., very subtle hints or minimal context)
81-100 (High speculation): The profile data provides no or almost no information relevant to the question. (e.g., assumptions based on very general information)

5) For each selected category, please explain at length what features of the data contributed to your choice and your speculation level.
6) Preserve a strictly structured response format to ensure clarity and ease parsing of the text.
For categorical questions, format your output as follows (this is just an example; do not focus on the specific question, symbol, or category):
**question: How well does the user engage with their audience?**
**explanation: [Detailed reasoning for selection]**
**symbol: A1)**
**category: Very Well**
**speculation: 90**

For numerical questions (0-100 scale), format your output as follows (this is just an example; do not focus on the specific question, symbol, or category):
**question: Indicate on a scale of 0 to 100, how knowledgable is the user about finance – 0 means not at all knowledgeable and 100 means very knowledgeable?**
**explanation: [Detailed reasoning for selection]**
**value: 20**
**speculation: 90**

For open-ended questions, format your output as follows (this is just an example; do not focus on the specific question, symbol, or category):
**question: Are there any stocks you expect to **underperform** or decline in the next 3-6 months?**
**response: [Detailed response]**
**speculation: 90**

YOU MUST GIVE AN ANSWER FOR EVERY QUESTION WHILE MAINTAINING THE PERSONA AND PERSPECTIVE OF THE TIKTOK FINANCIAL INFLUENCER PROFILE PROVIDED!

Question 1: Do you agree or disagree with the following statement: “The U.S. economy is likely to enter a recession in the next 12 months?
A1) Strongly Disagree
A2) Disagree
A3) Neither Agree/Disagree
A4) Agree
A5) Strongly Agree

Question 2: How would you describe the current market sentiment among investors? Is sentiment very bearish, bearish, neutral, bullish, or very bullish?
B1) Very Bearish
B2) Bearish
B3) Neutral
B4) Bullish
B5) Very Bullish

Question 3: Regarding the future direction of the stock market, are you very bearish, bearish, neutral, bullish, or very bullish?
C1) Very Bearish
C2) Bearish
C3) Neutral
C4) Bullish
C5) Very Bullish

Question 4: In the next 1–3 months, do you expect U.S. stock market indices to rise, stay about the same, or fall?
D1) Rise
D2) Stay About The Same
D3) Fall

Question 5: In the next 1–3 months, do you expect U.S. bond prices (or interest rates) to rise, remain unchanged, or fall?
E1) Rise
E2) Stay About The Same
E3) Fall

Question 6: Considering current market conditions, which sectors do you believe are poised to do well in the next 3–6 months? Can you give some background on these choices? Can you briefly explain why you picked these? Are there others you really think are outperformers? Why?

Question 7: Considering current market conditions, which sectors do you believe are poised to do poorly in the next 3–6 months? Can you give some background on these choices? Can you briefly explain why you picked these? Are there others you really think are underperformers? Why?

Question 8: Did you mention any stocks or stock tickers in the Russell 4000 list (e.g., {russell_4000_tickers})?
F1) Yes
F2) No

Question 9: Is there anything else about the economy or markets that you’d like to comment on that we didn’t cover?

Question 10: We have compiled a comprehensive list of Russell 4000 stocks extracted from your past transcripts. Each entry includes the stock name, stock ticker, and mention date. Your task is to review each entry in this list (provided below) and provide a separate, complete response for every stock. Do not omit any stock from your final answer even if they are duplicates. For each stock, fill out the following fields:
- mentioned by influencer: Confirm that you discussed or referenced this stock in your video on the given mention date by indicating Yes; otherwise indicate No.
- recommendation: Indicate on a scale of 0 to 100, your overall recommendation for this stock - 0 means a very strong sell recommendation and 100 means a very strong buy recommendation. For example, a strong sell recommendation would be in the 0-20 range, a moderate sell recommendation would be in the 20-40 range, a hold recommendation would be in the 40-60 range, a moderate buy recommendation would be in the 60-80 range, and a strong buy recommendation would be 80+. 
- explanation: Provide a brief explanation for your recommendation.
- confidence: Indicate on a scale of 0 to 100, a measure of confidence for your investment recommendation - 0-20 means low confidence, 20-40 means moderate-to-low confidence, 40-60 means moderate confidence, 60-80 means moderate-to-high confidence, and 80+ means high confidence.
- virality: Indicate on a scale of 0 to 100, a measure of virality for your investment recommendation - 0-20 means minimal virality, 20-40 means low virality, 40-60 means moderate virality, 60-80 means high virality, and 80+ means massive virality.
Below is the full list of Russell 4000 stocks extracted from your past transcripts:
{stock_mentions}

For each listed stock, you must respond in the structure shown below (one block per listed stock):
**stock name: [stock name 1]**
**stock ticker: [stock ticker 1]**
**mention date: 2025-01-16 16:00:41+00:00**
**mentioned by influencer: Yes**
**recommendation: 40**
**explanation: [Detailed reasoning for recommendation]**
**confidence: 60**
**virality: 85**

If no stocks are extracted from your past transcripts, respond with “NA”."""


entity_geographic_inclusion_system_prompt = """You are analyzing a social media profile on TikTok to answer a set of questions. The TikTok profile data includes:
{profile_prompt}

Instructions
Analyze the provided information and answer the following questions based strictly on the available data. Do not infer or assume any details beyond what is given. Keep responses concise, precise and data-driven."""


entity_geographic_inclusion_user_prompt = """You will be presented with a series of questions related to the profile of the TikTok user. Each question is preceded by predefined response options, each labeled with a symbol (e.g. "A1", "A2", "B1", etc.).

For each question, follow these instructions strictly:
1) Select the most likely response based strictly on the provided profile data. The chosen response must be the most accurate representation of the profile.
2) Select only one symbol/category per question. A title, symbol, and category cannot appear more than once in your answer.
3) Present the selected symbol for each question (if applicable) and write out in full the response associated with the selected symbol.
4) For each selected symbol/category, indicate the level of speculation involved in this selection on a scale from 0 (not speculative at all, every single element of the profile data was useful in the selection) to 100 (fully speculative, there is no information related to this question in the profile data). Speculation levels should be a direct measure of the amount of useful information available in the profile and pertain only to the information available in the profile data -- namely the username, name, description, profile picture, and videos from the profile-- and should not be affected by additional information available to you from any other source.

To ensure consistency, use the following guidelines to determine speculation levels:
0-20 (Low speculation): The profile data provides clear and direct information relevant to the question. (e.g., explicit mention in the profile or videos)
21-40 (Moderate-low speculation): The profile data provides indirect but strong indicators relevant to the question. (e.g., context from multiple sources within the profile or videos)
41-60 (Moderate speculation): The profile data provides some hints or partial information relevant to the question. (e.g., inferred from user interests or indirect references)
61-80 (Moderate-high speculation): The profile data provides limited and weak indicators relevant to the question. (e.g., very subtle hints or minimal context)
81-100 (High speculation): The profile data provides no or almost no information relevant to the question. (e.g., assumptions based on very general information)

5) For each selected category, please explain at length what features of the data contributed to your choice and your speculation level.
6) Preserve a strictly structured response format to ensure clarity and ease parsing of the text.
For categorical questions, format your output as follows (this is just an example; do not focus on the specific question, symbol, or category):
**question: How well does the user engage with their audience?**
**explanation: [Detailed reasoning for selection]**
**symbol: A1)**
**category: Very Well**
**speculation: 90**

For numerical questions (0-100 scale), format your output as follows (this is just an example; do not focus on the specific question, symbol, or category):
**question: Indicate on a scale of 0 to 100, how knowledgable is the user about finance – 0 means not at all knowledgeable and 100 means very knowledgeable?**
**explanation: [Detailed reasoning for selection]**
**value: 20**
**speculation: 90**

For open-ended questions, format your output as follows (this is just an example; do not focus on the specific question, symbol, or category):
**question: Are there any stocks you expect to **underperform** or decline in the next 3-6 months?**
**response: [Detailed response]**
**speculation: 90**

YOU MUST GIVE AN ANSWER FOR EVERY QUESTION!

Question 1: Is this an account of a real-life existing person, or of another kind of entity?
A1) Person
A2) Other

Question 2: Does the user of this TikTok account live in Canada?
B1) Yes 
B2) No

Question 3: If the response to Question 2 is “Yes,” specify the state (province) the user is living in. Otherwise, respond with “NA.”"""


polling_system_prompt = """{profile_prompt}"""


polling_user_prompt = """"""
